#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InvoiceGemini - Refactored application entry point with dependency injection.
Enhanced version incorporating all functionality from original main.py.
"""
import os
import sys
import logging
from pathlib import Path

# Setup HF_HUB_OFFLINE first (like in original main.py)
os.environ["HF_HUB_OFFLINE"] = "0"

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
        except (subprocess.SubprocessError, OSError, FileNotFoundError) as e:
            # Quietly ignore console encoding errors - not critical for operation
            pass
    except Exception as e:
        print(f"Failed to setup UTF-8 encoding: {e}")

# Import huggingface_hub early like in original
import huggingface_hub

# Set offline mode for transformers before imports
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Add project directory to sys.path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from PyQt6.QtWidgets import QApplication, QStyleFactory
from PyQt6.QtCore import Qt, QLocale, QTranslator

from app import config
from app import utils
from app.core.di_container import get_container, register_core_services
from app.settings_manager import settings_manager

# Import both window versions for compatibility
try:
    from app.ui.main_window_refactored import MainWindowRefactored
    REFACTORED_WINDOW_AVAILABLE = True
except ImportError as e:
    REFACTORED_WINDOW_AVAILABLE = False
    logging.warning(f"Refactored window not available: {e}")

from app.main_window import MainWindow


def setup_logging():
    """Setup application logging with enhanced debugging."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler with detailed format
    file_handler = logging.FileHandler(log_dir / 'app_debug.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler with simple format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Set specific loggers to WARNING to reduce noise
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return logger


def setup_portable_mode():
    """Setup paths for portable mode with enhanced logging."""
    logger = logging.getLogger(__name__)
    logger.info("DEBUG: Setting up portable mode...")
    
    # Path to application folder (using pathlib)
    app_dir = Path(__file__).parent
    
    # Create data folders if they don't exist
    data_dir = app_dir / "data"
    models_dir = data_dir / "models"
    temp_dir = data_dir / "temp"
    secrets_dir = data_dir / "secrets"
    
    for directory in [data_dir, models_dir, temp_dir, secrets_dir]:
        directory.mkdir(exist_ok=True, parents=True)
        logger.info(f"Directory created/verified: {directory}")
    
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Temp directory: {temp_dir}")
    
    # Load settings from file
    if hasattr(settings_manager, 'load_settings'):
        logger.info("DEBUG: Loading saved settings...")
        settings_manager.load_settings()
    
    # Update paths from settings
    logger.info("DEBUG: Updating tool paths...")
    try:
        config.update_paths_from_settings()
        logger.info(f"Poppler Path: {config.POPPLER_PATH}")
    except Exception as e:
        logger.warning(f"Error updating paths: {e}")
    
    # Setup Hugging Face token
    if config.HF_TOKEN:
        logger.info("DEBUG: Hugging Face token configured")
        os.environ["HF_TOKEN"] = config.HF_TOKEN
        # Set environment variable without login verification like in original
    else:
        logger.info("DEBUG: Hugging Face token not found")
    
    # Setup Google API key with enhanced error handling
    google_api_configured = False
    
    # Try secure storage first (DI container)
    container = get_container()
    try:
        secrets_manager = container.get('secrets_manager')
        google_api_key = secrets_manager.get_secret('google_api_key')
        if google_api_key:
            logger.info("DEBUG: Google API key retrieved from secure storage")
            import google.generativeai as genai
            genai.configure(api_key=google_api_key)
            logger.info("DEBUG: Google Gemini API initialized successfully from secure storage")
            google_api_configured = True
    except Exception as e:
        logger.warning(f"Failed to setup Google API from secure storage: {e}")
    
    # Fallback to config (like in original main.py)
    if not google_api_configured and config.GOOGLE_API_KEY:
        logger.info("DEBUG: Google API key configured from config")
        try:
            import google.generativeai as genai
            genai.configure(api_key=config.GOOGLE_API_KEY)
            logger.info("DEBUG: Google Gemini API successfully initialized from config")
        except ImportError:
            logger.warning("DEBUG: google-genai library not installed, Gemini will be unavailable")
        except Exception as e:
            logger.error(f"DEBUG: Error initializing Gemini API: {e}")
    elif not google_api_configured:
        logger.info("DEBUG: Google API key not found")
    
    # Setup offline mode
    if config.OFFLINE_MODE:
        logger.info("DEBUG: Offline mode enabled")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        logger.info("DEBUG: Offline mode disabled")
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"


def setup_application(app: QApplication):
    """Setup Qt application properties with enhanced logging."""
    logger = logging.getLogger(__name__)
    
    logger.info("DEBUG: Configuring QApplication...")
    app.setApplicationName(config.APP_NAME)
    app.setApplicationVersion(config.APP_VERSION)
    app.setOrganizationName(config.ORGANIZATION_NAME)
    logger.info("DEBUG: QApplication configured")
    
    # Setup application style
    logger.info("DEBUG: Setting Fusion style...")
    app.setStyle(QStyleFactory.create("Fusion"))
    logger.info("DEBUG: Style set")
    
    # Setup locale
    logger.info("DEBUG: Setting Russian locale...")
    locale = QLocale(QLocale.Language.Russian, QLocale.Country.Russia)
    QLocale.setDefault(locale)
    logger.info("DEBUG: Locale set")
    
    # Load translations
    logger.info("DEBUG: Setting up localization...")
    translator = QTranslator()
    locale_name = QLocale.system().name()
    if translator.load(f"translations/{locale_name}.qm"):
        app.installTranslator(translator)
        logger.info(f"DEBUG: Translation loaded for {locale_name}")
    else:
        logger.info("DEBUG: No translation file found, using default")
    logger.info("DEBUG: Localization configured")


def choose_main_window():
    """Choose between original and refactored main window."""
    logger = logging.getLogger(__name__)
    
    # Check if we should use refactored window
    use_refactored = REFACTORED_WINDOW_AVAILABLE and getattr(config, 'USE_REFACTORED_UI', True)
    
    if use_refactored:
        logger.info("DEBUG: Creating MainWindowRefactored...")
        try:
            main_window = MainWindowRefactored()
            logger.info("DEBUG: MainWindowRefactored created successfully")
            return main_window
        except Exception as e:
            logger.error(f"DEBUG: Failed to create MainWindowRefactored: {e}")
            logger.info("DEBUG: Falling back to original MainWindow...")
    
    # Fallback to original window
    logger.info("DEBUG: Creating original MainWindow...")
    main_window = MainWindow()
    logger.info("DEBUG: Original MainWindow created")
    return main_window


def main():
    """Application entry point with enhanced debugging and error handling."""
    # Setup logging first
    logger = setup_logging()
    logger.info("="*60)
    logger.info("DEBUG: Starting InvoiceGemini application...")
    logger.info(f"DEBUG: HF_HUB_OFFLINE forcibly set to 0 at start")
    
    try:
        # Initialize dependency injection container
        logger.info("DEBUG: Initializing DI container...")
        container = get_container()
        register_core_services(container)
        logger.info("DEBUG: Dependency injection container initialized")
        
        # Setup portable mode
        logger.info("DEBUG: Starting portable mode setup...")
        setup_portable_mode()
        logger.info("DEBUG: Portable mode configured")
        
        # Create Qt application
        logger.info("DEBUG: Creating QApplication...")
        app = QApplication(sys.argv)
        logger.info("DEBUG: QApplication created")
        
        # Setup application properties
        setup_application(app)
        
        # Create and show main window
        logger.info("DEBUG: Creating main window...")
        main_window = choose_main_window()
        
        logger.info("DEBUG: Showing main window...")
        main_window.show()
        logger.info("DEBUG: Main window shown")
        
        # Run event loop
        logger.info("DEBUG: Starting event loop...")
        logger.info("="*60)
        exit_code = app.exec()
        
        logger.info("DEBUG: Application finished")
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"CRITICAL: Application startup failed: {e}", exc_info=True)
        # Print to console as well for visibility
        print(f"CRITICAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 