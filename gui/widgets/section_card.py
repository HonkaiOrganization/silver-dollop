import os
import base64
import uuid
import logging

import cv2
import numpy as np
import markdown

from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QTextBrowser,
    QFrame, QProgressBar,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

logger = logging.getLogger(__name__)

TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "temp")


def _extract_keyframes(clip_b64: str, max_frames: int = 8) -> list[np.ndarray]:
    """Extract evenly-spaced keyframes from a base64-encoded video clip."""
    data = base64.b64decode(clip_b64)
    os.makedirs(TEMP_DIR, exist_ok=True)
    tmp_path = os.path.join(TEMP_DIR, f"card_clip_{uuid.uuid4().hex[:8]}.mp4")
    try:
        with open(tmp_path, 'wb') as f:
            f.write(data)
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []
        num = min(max_frames, total)
        step = max(1, total // num)
        frames = []
        for idx in range(0, total, step)[:num]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        return frames
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _create_mosaic(frames: list[np.ndarray], cols: int = 4, target_width: int = 960) -> np.ndarray | None:
    """Compose multiple frames into a grid mosaic image."""
    if not frames:
        return None
    rows = (len(frames) + cols - 1) // cols
    h, w = frames[0].shape[:2]
    cell_w = target_width // cols
    cell_h = int(h * cell_w / w)
    canvas = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        resized = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        canvas[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = resized
    return canvas


def _bgr_to_pixmap(bgr: np.ndarray) -> QPixmap:
    arr = np.ascontiguousarray(bgr)
    h, w, ch = arr.shape
    qimg = QImage(arr.data, w, h, ch * w, QImage.Format.Format_BGR888).copy()
    return QPixmap.fromImage(qimg)


class SectionCard(QFrame):
    def __init__(self, section: dict, index: int, total: int, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title_lbl = QLabel(section["title"])
        title_lbl.setStyleSheet("font-weight: bold; font-size: 14px;")
        header.addWidget(title_lbl)
        header.addStretch()

        prob = section["prob"]
        prob_lbl = QLabel(f"Abnormal Prob {prob:.1%}")
        header.addWidget(prob_lbl)

        frame_lbl = QLabel(f"Frame {section['start_frame']}-{section['end_frame']}")
        header.addWidget(frame_lbl)
        layout.addLayout(header)

        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(int(prob * 100))
        bar.setTextVisible(False)
        bar.setFixedHeight(6)
        layout.addWidget(bar)

        clip_b64 = section.get("clip_b64")
        if clip_b64:
            frames = _extract_keyframes(clip_b64, max_frames=8)
            mosaic = _create_mosaic(frames, cols=4, target_width=960)
            if mosaic is not None:
                fig_label = QLabel(f"Figure {section.get('figure_number', index)}")
                fig_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                fig_label.setStyleSheet("color: gray; font-size: 11px;")
                layout.addWidget(fig_label)

                mosaic_label = QLabel()
                pixmap = _bgr_to_pixmap(mosaic)
                mosaic_label.setPixmap(pixmap.scaled(
                    960, 600,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
                mosaic_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(mosaic_label)

        analysis_html = markdown.markdown(
            section["analysis"],
            extensions=["fenced_code", "tables", "sane_lists"],
        )
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(analysis_html)
        browser.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        browser.document().setTextWidth(600)
        doc_height = int(browser.document().size().height()) + 10
        browser.setFixedHeight(max(doc_height, 60))
        layout.addWidget(browser)

    def cleanup(self):
        pass
