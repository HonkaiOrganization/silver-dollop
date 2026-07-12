import os
import uuid
import logging

import numpy as np
import markdown
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib

matplotlib.use('Agg')

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextBrowser, QScrollArea, QProgressBar, QFrame, QFileDialog,
    QMessageBox,
)
from PySide6.QtCore import Qt, Signal

from core.infer import JumpRopeInference
from gui.workers import InferenceWorker, VLMWorker
from gui.widgets import SectionCard

logger = logging.getLogger(__name__)

TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp")


class AnalysisPage(QWidget):
    back_requested = Signal()

    def __init__(self, video_path: str, csv_path: str, parent=None):
        super().__init__(parent)
        self._video_path = video_path
        self._csv_path = csv_path
        self._inference = JumpRopeInference()
        self._infer_result: dict | None = None
        self._output_json = os.path.join(
            TEMP_DIR, f"infer_{uuid.uuid4().hex[:8]}.json"
        )
        self._inference_worker: InferenceWorker | None = None
        self._vlm_worker: VLMWorker | None = None
        self._section_cards: list[SectionCard] = []
        self._vlm_sections: list[dict] = []
        self._vlm_summary: str = ""
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        nav_bar = QHBoxLayout()
        nav_bar.setContentsMargins(0, 0, 0, 0)

        self.btn_back = QPushButton("Back to Recording")
        self.btn_back.clicked.connect(self.back_requested.emit)
        nav_bar.addWidget(self.btn_back)
        nav_bar.addStretch()

        self.lbl_status = QLabel("")
        nav_bar.addWidget(self.lbl_status)

        self.progress = QProgressBar()
        self.progress.setFixedWidth(200)
        self.progress.setRange(0, 0)
        self.progress.hide()
        nav_bar.addWidget(self.progress)

        root.addLayout(nav_bar)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 10, 0)
        self._content_layout.setSpacing(16)

        self.lbl_stats = QLabel()
        self.lbl_stats.setWordWrap(True)
        self.lbl_stats.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.lbl_stats.setFrameShape(QFrame.Shape.StyledPanel)
        self.lbl_stats.setContentsMargins(16, 16, 16, 16)
        self.lbl_stats.hide()
        self._content_layout.addWidget(self.lbl_stats)

        self._chart_container = QWidget()
        self._chart_layout = QVBoxLayout(self._chart_container)
        self._chart_layout.setContentsMargins(0, 0, 0, 0)
        self._chart_container.hide()
        self._content_layout.addWidget(self._chart_container)

        vlm_bar = QHBoxLayout()
        vlm_bar.setContentsMargins(0, 0, 0, 0)

        self.btn_vlm = QPushButton("VLM Deep Analysis")
        self.btn_vlm.setEnabled(False)
        self.btn_vlm.clicked.connect(self._start_vlm)
        vlm_bar.addWidget(self.btn_vlm)

        self.lbl_vlm_status = QLabel("")
        vlm_bar.addWidget(self.lbl_vlm_status)

        self.vlm_progress = QProgressBar()
        self.vlm_progress.setFixedWidth(200)
        self.vlm_progress.setRange(0, 0)
        self.vlm_progress.hide()
        vlm_bar.addWidget(self.vlm_progress)

        self.btn_export = QPushButton("Export Report")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._on_export_report)
        self.btn_export.hide()
        vlm_bar.addWidget(self.btn_export)

        vlm_bar.addStretch()

        self._vlm_widget = QWidget()
        self._vlm_widget.setLayout(vlm_bar)
        self._vlm_widget.hide()
        self._content_layout.addWidget(self._vlm_widget)

        self._sections_container = QWidget()
        self._sections_layout = QVBoxLayout(self._sections_container)
        self._sections_layout.setContentsMargins(0, 0, 0, 0)
        self._sections_layout.setSpacing(14)
        self._sections_container.hide()
        self._content_layout.addWidget(self._sections_container)

        self._summary_browser = QTextBrowser()
        self._summary_browser.setOpenExternalLinks(True)
        self._summary_browser.hide()
        self._content_layout.addWidget(self._summary_browser)

        self._content_layout.addStretch()
        self._scroll.setWidget(self._content)
        root.addWidget(self._scroll, 1)

    def start_analysis(self):
        self.lbl_stats.hide()
        self._chart_container.hide()
        self._vlm_widget.hide()
        self._sections_container.hide()
        self._summary_browser.hide()
        self.btn_vlm.setEnabled(False)
        self.lbl_vlm_status.setText("")
        self.lbl_status.setText("Analyzing…")
        self.progress.show()

        os.makedirs(TEMP_DIR, exist_ok=True)
        self._inference_worker = InferenceWorker(
            self._inference, self._csv_path, self._output_json
        )
        self._inference_worker.finished.connect(self._on_inference_done)
        self._inference_worker.start()

    def _on_inference_done(self, result: dict):
        self.progress.hide()
        self._infer_result = result

        csv_name = os.path.basename(self._csv_path)
        r = result["results"].get(csv_name)

        if not r or r.get("status") != "ok":
            err = r.get("reason", r.get("error", "Unknown error")) if r else "No results"
            self.lbl_status.setText(f"Analysis failed: {err}")
            return

        self.lbl_status.setText("Analysis complete")
        self._show_stats(r)
        self._plot_chart(r)
        self._vlm_widget.show()
        self.btn_vlm.setEnabled(True)

    def _show_stats(self, r: dict):
        label = r["predicted_label"]
        confidence = r["confidence"]
        p_normal = r["probabilities"]["normal"]
        p_abnormal = r["probabilities"]["abnormal"]
        num_windows = r["num_windows"]
        num_frames = r["num_frames"]

        label_en = "Normal" if label == "normal" else "Abnormal"

        self.lbl_stats.setText(
            f'Prediction: {label_en}\n'
            f'Confidence: {confidence:.4f}\n'
            f'P(Normal): {p_normal:.4f}  |  P(Abnormal): {p_abnormal:.4f}\n'
            f'Sliding Windows: {num_windows}  |  Total Frames: {num_frames}'
        )
        self.lbl_stats.show()

    def _plot_chart(self, r: dict):
        details = r.get("window_details", [])
        if not details:
            return

        for i in reversed(range(self._chart_layout.count())):
            w = self._chart_layout.itemAt(i).widget()  # type: ignore
            if w:
                w.deleteLater()

        fig = Figure(figsize=(8, 3.5), dpi=100)
        ax = fig.add_subplot(111)

        x = [d["start_frame"] for d in details]
        y = [d["prob_abnormal"] for d in details]

        ax.plot(x, y, marker='.', linewidth=1.2, markersize=4)
        ax.fill_between(x, y, alpha=0.15)
        ax.set_xlabel("Frame")
        ax.set_ylabel("P(abnormal)")
        ax.set_title("Per-Window Abnormal Confidence")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        canvas = FigureCanvas(fig)
        self._chart_layout.addWidget(canvas)
        self._chart_container.show()

    def _start_vlm(self):
        self.btn_vlm.setEnabled(False)
        self.lbl_vlm_status.setText("VLM analyzing…")
        self.vlm_progress.show()
        self._sections_container.hide()
        self._summary_browser.hide()

        self._vlm_worker = VLMWorker(
            self._video_path, self._output_json, top_k=3
        )
        self._vlm_worker.finished.connect(self._on_vlm_done)
        self._vlm_worker.error.connect(self._on_vlm_error)
        self._vlm_worker.start()

    def _on_vlm_done(self, result: dict):
        self.vlm_progress.hide()
        self.btn_vlm.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_export.show()
        self.lbl_vlm_status.setText("VLM analysis complete")
        self._vlm_sections = result.get("sections", [])
        self._vlm_summary = result.get("summary", "")
        self._render_sections(self._vlm_sections, self._vlm_summary)

    def _on_vlm_error(self, err: str):
        self.vlm_progress.hide()
        self.btn_vlm.setEnabled(True)
        self.lbl_vlm_status.setText(f"VLM failed: {err}")

    def _render_sections(self, sections: list[dict], summary: str):
        self._clear_sections()

        total = len(sections)
        for idx, sec in enumerate(sections):
            card = SectionCard(sec, idx + 1, total)
            self._section_cards.append(card)
            self._sections_layout.addWidget(card)

        if sections:
            self._sections_container.show()

        if summary:
            summary_html = markdown.markdown(
                summary, extensions=["fenced_code", "sane_lists"]
            )
            self._summary_browser.setHtml(
                f"<h3>总结与改进优先级</h3>{summary_html}"
            )
            self._summary_browser.document().setTextWidth(600)
            h = int(self._summary_browser.document().size().height()) + 20
            self._summary_browser.setFixedHeight(max(h, 60))
            self._summary_browser.show()

    def _clear_sections(self):
        for card in self._section_cards:
            card.cleanup()
            self._sections_layout.removeWidget(card)
            card.deleteLater()
        self._section_cards.clear()
        self._sections_container.hide()
        self._summary_browser.hide()

    def _on_export_report(self):
        if not self._vlm_sections:
            return

        default_name = os.path.join(
            os.path.dirname(self._video_path),
            "VLM_Analysis_Report.docx"
        )
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Export Report", default_name,
            "Word Document (*.docx)"
        )
        if not output_path:
            return

        try:
            from core.export_report import export_vlm_report
            csv_name = os.path.basename(self._csv_path)
            export_vlm_report(
                output_path=output_path,
                sections=self._vlm_sections,
                summary=self._vlm_summary,
                infer_result=self._infer_result,
                csv_name=csv_name,
            )
            QMessageBox.information(self, "Export Successful", f"Report saved to:\n{output_path}")
        except Exception as e:
            logger.exception("Failed to export report")
            QMessageBox.critical(self, "Export Failed", f"Error exporting report:\n{e}")

    def cleanup(self):
        if self._inference_worker and self._inference_worker.isRunning():
            self._inference_worker.abandon()
            self._inference_worker.quit()
            self._inference_worker.wait(2000)
        if self._vlm_worker and self._vlm_worker.isRunning():
            self._vlm_worker.quit()
            self._vlm_worker.wait(5000)
        self._clear_sections()
