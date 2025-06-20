"""
Image viewer widget component.
"""
from pathlib import Path
from typing import Optional, Union
import tempfile
import logging

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, 
    QScrollArea, QGroupBox, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QSize
from PyQt6.QtGui import QPixmap, QImage
from PIL import Image
import pdf2image


class ImageViewerSignals(QObject):
    """Signals for image viewer widget."""
    image_loaded = pyqtSignal(str)  # image path
    image_error = pyqtSignal(str)  # error message


class ImageViewerWidget(QWidget):
    """Widget for displaying images and PDFs."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.signals = ImageViewerSignals()
        
        self._current_image_path: Optional[Path] = None
        self._temp_dir = tempfile.TemporaryDirectory()
        
        self._init_ui()
        
    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout(self)
        
        # Create group box
        group_box = QGroupBox(self.tr("Изображение"))
        group_layout = QVBoxLayout()
        
        # Create scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        # Create image label
        self.image_label = QLabel(self.tr("Изображение не загружено"))
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.scroll_area.setWidget(self.image_label)
        
        group_layout.addWidget(self.scroll_area)
        group_box.setLayout(group_layout)
        layout.addWidget(group_box)
        
    def load_image(self, file_path: Union[str, Path]) -> bool:
        """Load and display image or PDF."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            if file_path.suffix.lower() == '.pdf':
                # Convert PDF to image
                image = self._convert_pdf_to_image(file_path)
            else:
                # Load image directly
                image = Image.open(file_path)
                
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Display image
            self._display_image(image)
            self._current_image_path = file_path
            
            self.signals.image_loaded.emit(str(file_path))
            self.logger.info(f"Image loaded: {file_path}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Failed to load image: {error_msg}")
            self.signals.image_error.emit(error_msg)
            self._show_error(self.tr("Ошибка загрузки: {0}").format(error_msg))
            return False
            
    def _convert_pdf_to_image(self, pdf_path: Path) -> Image.Image:
        """Convert first page of PDF to image."""
        try:
            # Try with custom poppler path first
            poppler_path = Path("resources/bin/poppler/poppler-24.08.0/Library/bin")
            if poppler_path.exists():
                images = pdf2image.convert_from_path(
                    pdf_path, 
                    dpi=200,
                    poppler_path=str(poppler_path)
                )
            else:
                # Fallback to system poppler
                images = pdf2image.convert_from_path(pdf_path, dpi=200)
                
            if not images:
                raise ValueError("No pages found in PDF")
                
            return images[0]
            
        except Exception as e:
            self.logger.error(f"PDF conversion error: {e}")
            raise
            
    def _display_image(self, image: Image.Image):
        """Display PIL image in the label."""
        # Convert PIL image to QPixmap
        image_bytes = image.tobytes('raw', 'RGB')
        qimage = QImage(
            image_bytes,
            image.width,
            image.height,
            image.width * 3,
            QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimage)
        
        # Scale pixmap to fit while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.scroll_area.size() * 0.95,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.adjustSize()
        
    def _show_error(self, message: str):
        """Show error message in the viewer."""
        self.image_label.setText(message)
        self.image_label.setStyleSheet("color: red;")
        
    def clear(self):
        """Clear the current image."""
        self._current_image_path = None
        self.image_label.clear()
        self.image_label.setText(self.tr("Изображение не загружено"))
        self.image_label.setStyleSheet("")
        
    def resize_image_to_fit(self):
        """Resize current image to fit the viewer."""
        if self._current_image_path and self.image_label.pixmap():
            pixmap = self.image_label.pixmap()
            scaled_pixmap = pixmap.scaled(
                self.scroll_area.size() * 0.95,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            
    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        if self._current_image_path:
            # Resize image when widget is resized
            self.resize_image_to_fit()
            
    @property
    def current_image_path(self) -> Optional[Path]:
        """Get current image path."""
        return self._current_image_path
        
    @property
    def has_image(self) -> bool:
        """Check if an image is loaded."""
        return self._current_image_path is not None
        
    def __del__(self):
        """Cleanup temporary directory."""
        try:
            self._temp_dir.cleanup()
        except (OSError, AttributeError) as e:
            # Temporary directory может быть уже очищен или недоступен
            pass 