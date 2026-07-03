import os
import base64
import tempfile
import uuid

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import markdown

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextBrowser, QScrollArea, QSizePolicy, QProgressBar,
    QFrame, QSlider, QStyle,
)
from PySide6.QtCore import Qt, QThread, Signal, QUrl, QBuffer, QByteArray, QIODevice
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget

from core.infer import JumpRopeInference
from core.vlm import analyze_windows

TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp")


# ═══════════════════════════════════════════════════════════════════════
# Worker threads
# ═══════════════════════════════════════════════════════════════════════

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
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, video_path: str, json_path: str, top_k: int = 3):
        super().__init__()
        self.video_path = video_path
        self.json_path = json_path
        self.top_k = top_k

    def run(self):
        try:
            result = analyze_windows(
                self.video_path, self.json_path, top_k=self.top_k,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ═══════════════════════════════════════════════════════════════════════
# Video player widget
# ═══════════════════════════════════════════════════════════════════════

class VideoPlayer(QWidget):
    def __init__(self, clip_b64: str, parent=None):
        super().__init__(parent)
        self._tmp_file: str | None = None
        self._setup_ui()
        self._load_clip(clip_b64)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._video_widget = QVideoWidget()
        self._video_widget.setMinimumHeight(180)
        self._video_widget.setStyleSheet("background:#000;border-radius:6px")
        layout.addWidget(self._video_widget, 1)

        controls = QHBoxLayout()
        controls.setContentsMargins(4, 0, 4, 0)

        self._btn_play = QPushButton()
        self._btn_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self._btn_play.setFixedSize(32, 28)
        self._btn_play.setStyleSheet(
            "QPushButton{background:#333;border-radius:4px;border:none}"
            "QPushButton:hover{background:#555}"
        )
        self._btn_play.clicked.connect(self._toggle_play)
        controls.addWidget(self._btn_play)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 1000)
        self._slider.sliderMoved.connect(self._seek)
        self._slider.setStyleSheet("""
            QSlider::groove:horizontal {height:4px;background:#444;border-radius:2px}
            QSlider::handle:horizontal {background:#8e44ad;width:12px;height:12px;
                margin:-4px 0;border-radius:6px}
            QSlider::sub-page:horizontal {background:#8e44ad;border-radius:2px}
        """)
        controls.addWidget(self._slider, 1)

        self._lbl_time = QLabel("0:00")
        self._lbl_time.setStyleSheet("color:#888;font-size:11px")
        self._lbl_time.setFixedWidth(40)
        controls.addWidget(self._lbl_time)

        layout.addLayout(controls)

        self._player = QMediaPlayer()
        self._audio = QAudioOutput()
        self._audio.setMuted(True)
        self._player.setAudioOutput(self._audio)
        self._player.setVideoOutput(self._video_widget)
        self._player.positionChanged.connect(self._on_position)
        self._player.durationChanged.connect(self._on_duration)

    def _load_clip(self, clip_b64: str):
        try:
            data = base64.b64decode(clip_b64)
            fd, path = tempfile.mkstemp(suffix='.mp4', dir=TEMP_DIR)
            os.write(fd, data)
            os.close(fd)
            self._tmp_file = path
            self._player.setSource(QUrl.fromLocalFile(path))
        except Exception:
            pass

    def _toggle_play(self):
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
            self._btn_play.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        else:
            self._player.play()
            self._btn_play.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))

    def _seek(self, pos: int):
        duration = self._player.duration()
        if duration > 0:
            self._player.setPosition(int(pos * duration / 1000))

    def _on_position(self, pos: int):
        duration = self._player.duration()
        if duration > 0:
            self._slider.blockSignals(True)
            self._slider.setValue(int(pos * 1000 / duration))
            self._slider.blockSignals(False)
        secs = pos // 1000
        self._lbl_time.setText(f"{secs // 60}:{secs % 60:02d}")

    def _on_duration(self, duration: int):
        self._slider.setRange(0, 1000)

    def stop(self):
        self._player.stop()

    def cleanup(self):
        self._player.stop()
        if self._tmp_file and os.path.exists(self._tmp_file):
            try:
                os.unlink(self._tmp_file)
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════════════════
# Section card widget
# ═══════════════════════════════════════════════════════════════════════

class SectionCard(QFrame):
    def __init__(self, section: dict, index: int, total: int, parent=None):
        super().__init__(parent)
        self._video_player: VideoPlayer | None = None
        self.setStyleSheet("""
            SectionCard {
                background: #252525;
                border: 1px solid #333;
                border-radius: 10px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        # ── Header ─────────────────────────────────────────────────────
        header = QHBoxLayout()

        title_lbl = QLabel(section["title"])
        title_lbl.setStyleSheet(
            "color:#eee;font-size:16px;font-weight:bold;background:transparent;border:none"
        )
        header.addWidget(title_lbl)
        header.addStretch()

        prob = section["prob"]
        color = "#e74c3c" if prob > 0.7 else "#f39c12" if prob > 0.5 else "#27ae60"
        prob_lbl = QLabel(f" 异常概率 {prob:.1%} ")
        prob_lbl.setStyleSheet(
            f"background:{color};color:#fff;font-size:12px;font-weight:bold;"
            f"border-radius:10px;padding:3px 10px"
        )
        header.addWidget(prob_lbl)

        frame_lbl = QLabel(f"帧 {section['start_frame']}-{section['end_frame']}")
        frame_lbl.setStyleSheet("color:#888;font-size:12px;background:transparent;border:none")
        header.addWidget(frame_lbl)

        layout.addLayout(header)

        # ── Progress bar ───────────────────────────────────────────────
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(int(prob * 100))
        bar.setTextVisible(False)
        bar.setFixedHeight(6)
        bar.setStyleSheet(f"""
            QProgressBar {{background:#2a2a4e;border-radius:3px;border:none}}
            QProgressBar::chunk {{background:{color};border-radius:3px}}
        """)
        layout.addWidget(bar)

        # ── Video player ───────────────────────────────────────────────
        clip_b64 = section.get("clip_b64")
        if clip_b64:
            self._video_player = VideoPlayer(clip_b64)
            layout.addWidget(self._video_player)

        # ── Analysis text ──────────────────────────────────────────────
        analysis_html = markdown.markdown(
            section["analysis"],
            extensions=["fenced_code", "tables", "sane_lists"],
        )
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(f"""<!DOCTYPE html>
<html><head><style>
body {{background:transparent;color:#ddd;font-size:13px;line-height:1.7;margin:0}}
h1,h2,h3,h4 {{color:#eee;margin-top:8px}}
strong {{color:#f0c040}}
ul,ol {{padding-left:20px}}
li {{margin:3px 0}}
code {{background:#2a2a4e;padding:1px 5px;border-radius:3px;font-size:12px}}
pre {{background:#2a2a4e;padding:10px;border-radius:6px;overflow-x:auto}}
hr {{border:none;border-top:1px solid #333;margin:10px 0}}
</style></head><body>{analysis_html}</body></html>""")
        browser.setStyleSheet(
            "QTextBrowser{background:transparent;border:none;padding:0}"
        )
        browser.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        browser.document().setTextWidth(600)
        doc_height = int(browser.document().size().height()) + 10
        browser.setFixedHeight(max(doc_height, 60))
        layout.addWidget(browser)

    def cleanup(self):
        if self._video_player:
            self._video_player.cleanup()


# ═══════════════════════════════════════════════════════════════════════
# Analysis page
# ═══════════════════════════════════════════════════════════════════════

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
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ── Top nav bar ────────────────────────────────────────────────
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

        # ── Scroll area ───────────────────────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(
            "QScrollArea{border:none;background:#1e1e1e}"
        )

        self._content = QWidget()
        self._content.setStyleSheet("background:#1e1e1e")
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 10, 0)
        self._content_layout.setSpacing(16)

        # Stats card
        self.lbl_stats = QLabel()
        self.lbl_stats.setWordWrap(True)
        self.lbl_stats.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.lbl_stats.setStyleSheet(
            "QLabel{background:#252525;padding:16px;border-radius:8px;"
            "border:1px solid #333;color:#eee;font-size:14px}"
        )
        self.lbl_stats.hide()
        self._content_layout.addWidget(self.lbl_stats)

        # Chart container
        self._chart_container = QWidget()
        self._chart_layout = QVBoxLayout(self._chart_container)
        self._chart_layout.setContentsMargins(0, 0, 0, 0)
        self._chart_container.hide()
        self._content_layout.addWidget(self._chart_container)

        # VLM bar
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

        self._vlm_widget = QWidget()
        self._vlm_widget.setLayout(vlm_bar)
        self._vlm_widget.hide()
        self._content_layout.addWidget(self._vlm_widget)

        # Section cards container
        self._sections_container = QWidget()
        self._sections_layout = QVBoxLayout(self._sections_container)
        self._sections_layout.setContentsMargins(0, 0, 0, 0)
        self._sections_layout.setSpacing(14)
        self._sections_container.hide()
        self._content_layout.addWidget(self._sections_container)

        # Summary card (rendered after sections)
        self._summary_browser = QTextBrowser()
        self._summary_browser.setOpenExternalLinks(True)
        self._summary_browser.setStyleSheet(
            "QTextBrowser{background:#252525;color:#ddd;border:1px solid #333;"
            "border-radius:8px;padding:12px;font-size:13px}"
        )
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
            w = self._chart_layout.itemAt(i).widget() # type: ignore
            if w:
                w.deleteLater()

        fig = Figure(figsize=(8, 3.5), dpi=100, facecolor='#252525')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#252525')

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
        self.lbl_vlm_status.setText("VLM 分析完成")
        self._render_sections(result.get("sections", []), result.get("summary", ""))

    def _on_vlm_error(self, err: str):
        self.vlm_progress.hide()
        self.btn_vlm.setEnabled(True)
        self.lbl_vlm_status.setText(f"VLM 失败: {err}")

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
            self._summary_browser.setHtml(f"""<!DOCTYPE html>
<html><head><style>
body {{background:transparent;color:#ddd;font-size:13px;line-height:1.7;margin:0}}
strong {{color:#f0c040}}
</style></head><body>
<h3 style="color:#eee;margin-top:0">📋 总结与改进优先级</h3>
{summary_html}
</body></html>""")
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

    # ──────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────
    def cleanup(self):
        if self._inference_worker and self._inference_worker.isRunning():
            self._inference_worker.abandon()
            self._inference_worker.quit()
            self._inference_worker.wait(2000)
        if self._vlm_worker and self._vlm_worker.isRunning():
            self._vlm_worker.quit()
            self._vlm_worker.wait(5000)
        self._clear_sections()
