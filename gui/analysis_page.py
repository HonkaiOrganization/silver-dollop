import os
import json
import uuid

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import markdown

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextBrowser, QScrollArea, QSizePolicy, QProgressBar, QApplication
)
from PySide6.QtCore import Qt, QThread, Signal

from core.infer import JumpRopeInference
from core.vlm import analyze_windows

TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp")


class InferenceWorker(QThread):
    finished = Signal(dict)

    def __init__(self, inference: JumpRopeInference, csv_path: str, output_json: str):
        super().__init__()
        self._inference = inference
        self._csv_path = csv_path
        self._output_json = output_json
        self._abandoned = False

    def abandon(self):
        self._abandoned = True

    def run(self):
        result = self._inference.predict(self._csv_path, self._output_json)
        if not self._abandoned:
            self.finished.emit(result)


class VLMWorker(QThread):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, video_path: str, json_path: str, top_k: int = 3):
        super().__init__()
        self.video_path = video_path
        self.json_path = json_path
        self.top_k = top_k

    def run(self):
        try:
            report = analyze_windows(self.video_path, self.json_path, top_k=self.top_k)
            self.finished.emit(report)
        except Exception as e:
            self.error.emit(str(e))


def _md_to_html(md: str) -> str:
    return markdown.markdown(
        md,
        extensions=["fenced_code", "tables", "sane_lists"],
        output_format="html"
    )


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
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        nav_bar = QHBoxLayout()
        nav_bar.setContentsMargins(0, 0, 0, 0)

        self.btn_back = QPushButton("← 返回录制")
        self.btn_back.setStyleSheet(
            "QPushButton{background:#555;color:#fff;border-radius:4px;padding:6px 18px}"
            "QPushButton:hover{background:#666}"
        )
        self.btn_back.clicked.connect(self.back_requested.emit)
        nav_bar.addWidget(self.btn_back)
        nav_bar.addStretch()

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color:#aaa;padding-left:12px")
        nav_bar.addWidget(self.lbl_status)

        self.progress = QProgressBar()
        self.progress.setFixedWidth(200)
        self.progress.setRange(0, 0)
        self.progress.hide()
        nav_bar.addWidget(self.progress)

        root.addLayout(nav_bar)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{border:none}")

        content = QWidget()
        self._content_layout = QVBoxLayout(content)
        self._content_layout.setContentsMargins(0, 0, 10, 0)
        self._content_layout.setSpacing(16)

        self.lbl_stats = QLabel()
        self.lbl_stats.setWordWrap(True)
        self.lbl_stats.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.lbl_stats.setStyleSheet(
            "QLabel{background:#1a1a2e;padding:16px;border-radius:8px;"
            "color:#eee;font-size:14px}"
        )
        self.lbl_stats.hide()
        self._content_layout.addWidget(self.lbl_stats)

        self._chart_container = QWidget()
        self._chart_layout = QVBoxLayout(self._chart_container)
        self._chart_layout.setContentsMargins(0, 0, 0, 0)
        self._chart_container.hide()
        self._content_layout.addWidget(self._chart_container)

        vlm_bar = QHBoxLayout()
        vlm_bar.setContentsMargins(0, 0, 0, 0)

        self.btn_vlm = QPushButton("🔍 VLM 深度分析")
        self.btn_vlm.setStyleSheet(
            "QPushButton{background:#8e44ad;color:#fff;border-radius:4px;padding:8px 24px;font-size:14px}"
            "QPushButton:hover{background:#9b59b6}"
            "QPushButton:disabled{background:#555;color:#888}"
        )
        self.btn_vlm.setEnabled(False)
        self.btn_vlm.clicked.connect(self._start_vlm)
        vlm_bar.addWidget(self.btn_vlm)

        self.lbl_vlm_status = QLabel("")
        self.lbl_vlm_status.setStyleSheet("color:#aaa;padding-left:12px")
        vlm_bar.addWidget(self.lbl_vlm_status)

        self.vlm_progress = QProgressBar()
        self.vlm_progress.setFixedWidth(200)
        self.vlm_progress.setRange(0, 0)
        self.vlm_progress.hide()
        vlm_bar.addWidget(self.vlm_progress)
        vlm_bar.addStretch()

        vlm_widget = QWidget()
        vlm_widget.setLayout(vlm_bar)
        vlm_widget.hide()
        self._vlm_widget = vlm_widget
        self._content_layout.addWidget(vlm_widget)

        self.report_browser = QTextBrowser()
        self.report_browser.setOpenExternalLinks(True)
        self.report_browser.setStyleSheet(
            "QTextBrowser{background:#1e1e2e;color:#eee;border:1px solid #333;"
            "border-radius:6px;padding:12px;font-size:14px}"
        )
        self.report_browser.hide()
        self._content_layout.addWidget(self.report_browser)

        self._content_layout.addStretch()
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

    def start_analysis(self):
        """清空之前的结果，启动推理"""
        self.lbl_stats.hide()
        self._chart_container.hide()
        self._vlm_widget.hide()
        self.report_browser.hide()
        self.btn_vlm.setEnabled(False)
        self.lbl_vlm_status.setText("")
        self.lbl_status.setText("正在分析…")
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
            err = r.get("reason", r.get("error", "未知错误")) if r else "无结果"
            self.lbl_status.setText(f"分析失败: {err}")
            return

        self.lbl_status.setText("分析完成")
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

        label_cn = "正常" if label == "normal" else "异常"
        color = "#27ae60" if label == "normal" else "#e74c3c"

        self.lbl_stats.setText(
            f'<span style="font-size:20px;font-weight:bold">判定结果: '
            f'<span style="color:{color}">{label_cn}</span></span><br><br>'
            f'置信度: <b>{confidence:.4f}</b><br>'
            f'P(正常): {p_normal:.4f}  |  P(异常): {p_abnormal:.4f}<br>'
            f'滑动窗口数: {num_windows}  |  总帧数: {num_frames}'
        )
        self.lbl_stats.show()

    def _plot_chart(self, r: dict):
        details = r.get("window_details", [])
        if not details:
            return

        for i in reversed(range(self._chart_layout.count())):
            item = self._chart_layout.itemAt(i)
            if item is None:
                continue
            w = item.widget()
            if w:
                w.deleteLater()

        fig = Figure(figsize=(8, 3.5), dpi=100, facecolor='#1a1a2e')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#1a1a2e')

        x = [d["start_frame"] for d in details]
        y = [d["prob_abnormal"] for d in details]

        ax.plot(x, y, color='#e74c3c', marker='.', linewidth=1.2, markersize=4)
        ax.fill_between(x, y, alpha=0.15, color='#e74c3c')
        ax.set_xlabel("Frame", color='#aaa', fontsize=10)
        ax.set_ylabel("P(abnormal)", color='#aaa', fontsize=10)
        ax.set_title("逐窗口异常置信度", color='#eee', fontsize=13, pad=10)
        ax.set_ylim(0, 1.05)
        ax.tick_params(colors='#888')
        ax.grid(True, alpha=0.2, color='#555')
        for spine in ax.spines.values():
            spine.set_color('#444')

        fig.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.setStyleSheet("background:transparent")
        self._chart_layout.addWidget(canvas)
        self._chart_container.show()

    def _start_vlm(self):
        self.btn_vlm.setEnabled(False)
        self.lbl_vlm_status.setText("VLM 分析中…")
        self.vlm_progress.show()
        self.report_browser.hide()

        self._vlm_worker = VLMWorker(
            self._video_path, self._output_json, top_k=3
        )
        self._vlm_worker.finished.connect(self._on_vlm_done)
        self._vlm_worker.error.connect(self._on_vlm_error)
        self._vlm_worker.start()

    def _on_vlm_done(self, report: str):
        self.vlm_progress.hide()
        self.btn_vlm.setEnabled(True)
        self.lbl_vlm_status.setText("VLM 分析完成")
        self.report_browser.setHtml(_md_to_html(report))
        self.report_browser.show()

    def _on_vlm_error(self, err: str):
        self.vlm_progress.hide()
        self.btn_vlm.setEnabled(True)
        self.lbl_vlm_status.setText(f"VLM 失败: {err}")

    def cleanup(self):
        if self._inference_worker and self._inference_worker.isRunning():
            self._inference_worker.abandon()
            self._inference_worker.quit()
            self._inference_worker.wait(2000)
        if self._vlm_worker and self._vlm_worker.isRunning():
            self._vlm_worker.quit()
            self._vlm_worker.wait(5000)
