import markdown

from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QTextBrowser,
    QFrame, QProgressBar,
)
from PySide6.QtCore import Qt

from gui.widgets.video_player import VideoPlayer


class SectionCard(QFrame):
    def __init__(self, section: dict, index: int, total: int, parent=None):
        super().__init__(parent)
        self._video_player: VideoPlayer | None = None
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
        if prob > 0.7:
            prob_text = f"Abnormal Prob {prob:.1%}"
        elif prob > 0.5:
            prob_text = f"Abnormal Prob {prob:.1%}"
        else:
            prob_text = f"Abnormal Prob {prob:.1%}"
        prob_lbl = QLabel(prob_text)
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
            self._video_player = VideoPlayer(clip_b64)
            layout.addWidget(self._video_player)

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
        if self._video_player:
            self._video_player.cleanup()
