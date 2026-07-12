import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap


class FrameDisplay(QWidget):
    """Reusable widget that displays a BGR numpy frame with a title label."""

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._last_pixmap: QPixmap | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._title_label = QLabel(title)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet("background-color: black;")
        self._image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._image_label.setMinimumSize(1, 1)

        layout.addWidget(self._title_label)
        layout.addWidget(self._image_label, 1)

    @property
    def title(self) -> str:
        return self._title_label.text()

    @title.setter
    def title(self, text: str):
        self._title_label.setText(text)

    @property
    def image_label(self) -> QLabel:
        return self._image_label

    def update_frame(self, bgr: np.ndarray):
        arr = np.ascontiguousarray(bgr)
        h, w, ch = arr.shape
        qimg = QImage(arr.data, w, h, ch * w, QImage.Format.Format_BGR888).copy()
        pixmap = QPixmap.fromImage(qimg)
        self._last_pixmap = pixmap
        self._image_label.setPixmap(pixmap.scaled(
            self._image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def clear(self):
        self._image_label.clear()
        self._last_pixmap = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._last_pixmap:
            self._image_label.setPixmap(self._last_pixmap.scaled(
                self._image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
