#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InvoiceGemini - Refactored application entry point with dependency injection.
"""
import os
import sys
import logging
from pathlib import Path

# Setup UTF-8 encoding for Windows
if sys.platform == "win32":
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
        
        # Try to change Windows console encoding to UTF-8
        import subprocess
        try:
            subprocess.run(['chcp', '65001'], capture_output=True, check=False)
        except:
            pass
    except Exception as e:
        print(f"Failed to setup UTF-8 encoding: {e}")

# Add project directory to sys.path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from PyQt6.QtWidgets import QApplication, QStyleFactory
from PyQt6.QtCore import Qt, QLocale, QTranslator

from app import config
from app.core.di_container import get_container, register_core_services
from app.ui.main_window_refactored import MainWindowRefactored


def setup_logging():
    """Setup application logging."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'app.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific loggers to WARNING to reduce noise
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('huggingface_hub').setLevel(logging.WARNING)


def setup_portable_mode():
    """Setup paths for portable mode."""
    logger = logging.getLogger(__name__)
    logger.info("Setting up portable mode...")
    
    # Create necessary directories
    app_dir = Path(__file__).parent
    data_dir = app_dir / "data"
    models_dir = data_dir / "models"
    temp_dir = data_dir / "temp"
    secrets_dir = data_dir / "secrets"
    
    for directory in [data_dir, models_dir, temp_dir, secrets_dir]:
        directory.mkdir(exist_ok=True, parents=True)
        logger.info(f"Created directory: {directory}")
    
    # Load settings
    from app.settings_manager import settings_manager
    if hasattr(settings_manager, 'load_settings'):
        logger.info("Loading saved settings...")
        settings_manager.load_settings()
    
    # Update paths from settings
    try:
        config.update_paths_from_settings()
        logger.info(f"Poppler Path: {config.POPPLER_PATH}")
    except Exception as e:
        logger.warning(f"Error updating paths: {e}")
    
    # Setup Hugging Face token
    if config.HF_TOKEN:
        logger.info("Hugging Face token configured")
        os.environ["HF_TOKEN"] = config.HF_TOKEN
    else:
        logger.info("Hugging Face token not found")
    
    # Setup Google API key (using secure storage if available)
    container = get_container()
    try:
        secrets_manager = container.get('secrets_manager')
        google_api_key = secrets_manager.get_secret('google_api_key')
        if google_api_key:
            logger.info("Google API key retrieved from secure storage")
            import google.generativeai as genai
            genai.configure(api_key=google_api_key)
            logger.info("Google Gemini API initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to setup Google API from secure storage: {e}")
        # Fallback to config
        if config.GOOGLE_API_KEY:
            logger.info("Using Google API key from config")
            try:
                import google.generativeai as genai
                genai.configure(api_key=config.GOOGLE_API_KEY)
                logger.info("Google Gemini API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini API: {e}")
    
    # Setup offline mode
    if config.OFFLINE_MODE:
        logger.info("Offline mode enabled")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        logger.info("Offline mode disabled")
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"


def setup_application(app: QApplication):
    """Setup Qt application properties."""
    app.setApplicationName(config.APP_NAME)
    app.setApplicationVersion(config.APP_VERSION)
    app.setOrganizationName(config.ORGANIZATION_NAME)
    
    # Setup application style
    app.setStyle(QStyleFactory.create("Fusion"))
    
    # Setup locale
    locale = QLocale(QLocale.Language.Russian, QLocale.Country.Russia)
    QLocale.setDefault(locale)
    
    # Load translations
    translator = QTranslator()
    locale_name = QLocale.system().name()
    if translator.load(f"translations/{locale_name}.qm"):
        app.installTranslator(translator)


def main():
    """Application entry point."""
    # Setup logging first
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting InvoiceGemini application...")
    
    # Initialize dependency injection container
    container = get_container()
    register_core_services(container)
    logger.info("Dependency injection container initialized")
    
    # Setup portable mode
    setup_portable_mode()
    logger.info("Portable mode configured")
    
    # Create Qt application
    app = QApplication(sys.argv)
    setup_application(app)
    logger.info("Qt application created and configured")
    
    # Create and show main window
    try:
        main_window = MainWindowRefactored()
        main_window.show()
        logger.info("Main window created and shown")
    except Exception as e:
        logger.error(f"Failed to create main window: {e}", exc_info=True)
        raise
    
    # Run event loop
    logger.info("Starting event loop...")
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 