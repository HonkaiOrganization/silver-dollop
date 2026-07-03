import os
import uuid
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QComboBox, QSizePolicy,
    QPushButton, QSlider, QMessageBox, QStackedWidget
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

from models.camera import CameraManager
from models.pose import PoseProcessor
from gui.camera_thread import CameraThread
from gui.playback_thread import PlaybackThread
from gui.analysis_page import AnalysisPage

TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.camera_manager = CameraManager()
        self.pose_processor = PoseProcessor()
        self.camera_thread = None
        self.playback_thread = None

        self._last_camera_pixmap = None
        self._last_skeleton_pixmap = None
        self._mode = "recording"  # "recording" | "playback"

        self.init_ui()
        self.init_camera_system()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------
    def init_ui(self):
        self.setWindowTitle("跳绳姿态录制与分析")

        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        width = int(screen_geometry.width() * 0.8)
        height = int(width * 9 / 16)
        self.resize(width, height)

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        # ── 主页（录制 / 回放）──────────────────────────────────────────
        self._main_page = QWidget()
        main_layout = QVBoxLayout(self._main_page)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)
        self._stack.addWidget(self._main_page)

        self._analysis_page: AnalysisPage | None = None

        # ── 录制模式控制栏 ──────────────────────────────────────────────
        self.recording_bar = QWidget()
        rec_layout = QHBoxLayout(self.recording_bar)
        rec_layout.setContentsMargins(0, 0, 0, 0)

        self.camera_selector = QComboBox()
        self.camera_selector.setMinimumWidth(150)
        self.camera_selector.currentIndexChanged.connect(self._on_camera_switched)

        self.btn_start_rec = QPushButton("● 开始记录")
        self.btn_start_rec.setStyleSheet(
            "QPushButton{background:#c0392b;color:#fff;border-radius:4px;padding:6px 18px}"
            "QPushButton:hover{background:#e74c3c}"
        )
        self.btn_start_rec.clicked.connect(self._on_start_recording)

        self.btn_stop_rec = QPushButton("■ 结束记录")
        self.btn_stop_rec.setEnabled(False)
        self.btn_stop_rec.setStyleSheet(
            "QPushButton{background:#555;color:#fff;border-radius:4px;padding:6px 18px}"
            "QPushButton:enabled{background:#2980b9}"
            "QPushButton:enabled:hover{background:#3498db}"
        )
        self.btn_stop_rec.clicked.connect(self._on_stop_recording)

        self.lbl_rec_status = QLabel("")
        self.lbl_rec_status.setStyleSheet("color:#aaa;padding-left:12px")

        rec_layout.addWidget(QLabel("摄像机:"))
        rec_layout.addWidget(self.camera_selector)
        rec_layout.addSpacing(20)
        rec_layout.addWidget(self.btn_start_rec)
        rec_layout.addWidget(self.btn_stop_rec)
        rec_layout.addWidget(self.lbl_rec_status)
        rec_layout.addStretch()

        # ── 回放模式控制栏 ──────────────────────────────────────────────
        self.playback_bar = QWidget()
        pb_layout = QHBoxLayout(self.playback_bar)
        pb_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_play_pause = QPushButton("▶ 播放")
        self.btn_play_pause.setFixedWidth(90)
        self.btn_play_pause.setStyleSheet(
            "QPushButton{background:#27ae60;color:#fff;border-radius:4px;padding:6px 0}"
            "QPushButton:hover{background:#2ecc71}"
        )
        self.btn_play_pause.clicked.connect(self._on_play_pause)

        self.slider_progress = QSlider(Qt.Orientation.Horizontal)
        self.slider_progress.setMinimum(0)
        self.slider_progress.setMaximum(0)
        self.slider_progress.sliderMoved.connect(self._on_slider_moved)
        self.slider_progress.sliderPressed.connect(self._on_slider_pressed)
        self.slider_progress.sliderReleased.connect(self._on_slider_released)
        self._slider_dragging = False

        self.lbl_frame_info = QLabel("0 / 0")
        self.lbl_frame_info.setFixedWidth(110)
        self.lbl_frame_info.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_frame_info.setStyleSheet("color:#aaa")

        self.btn_submit = QPushButton("提交分析")
        self.btn_submit.setStyleSheet(
            "QPushButton{background:#8e44ad;color:#fff;border-radius:4px;padding:6px 20px}"
            "QPushButton:hover{background:#9b59b6}"
        )
        self.btn_submit.clicked.connect(self._on_submit_analysis)

        pb_layout.addWidget(self.btn_play_pause)
        pb_layout.addSpacing(10)
        pb_layout.addWidget(self.slider_progress, 1)
        pb_layout.addWidget(self.lbl_frame_info)
        pb_layout.addSpacing(20)
        pb_layout.addWidget(self.btn_submit)

        self.playback_bar.hide()

        # ── 双路图像视图 ────────────────────────────────────────────────
        views_layout = QHBoxLayout()
        views_layout.setSpacing(10)

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.camera_view = QLabel()
        self.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_view.setStyleSheet("background-color: #1e1e1e; border: 1px solid #333;")
        self.camera_view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.camera_view.setMinimumSize(1, 1)

        self.lbl_camera_title = QLabel("Camera Feed")
        self.lbl_camera_title.setStyleSheet("color: #888; padding: 4px;")
        self.lbl_camera_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        left_layout.addWidget(self.lbl_camera_title)
        left_layout.addWidget(self.camera_view, 1)

        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.skeleton_view = QLabel()
        self.skeleton_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.skeleton_view.setStyleSheet("background-color: #000000; border: 1px solid #333;")
        self.skeleton_view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.skeleton_view.setMinimumSize(1, 1)

        self.lbl_skeleton_title = QLabel("Pose Skeleton")
        self.lbl_skeleton_title.setStyleSheet("color: #888; padding: 4px;")
        self.lbl_skeleton_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        right_layout.addWidget(self.lbl_skeleton_title)
        right_layout.addWidget(self.skeleton_view, 1)

        views_layout.addWidget(left_container, 1)
        views_layout.addWidget(right_container, 1)

        main_layout.addWidget(self.recording_bar)
        main_layout.addWidget(self.playback_bar)
        main_layout.addLayout(views_layout, 1)

    # ------------------------------------------------------------------
    # 摄像机初始化
    # ------------------------------------------------------------------
    def init_camera_system(self):
        cameras = self.camera_manager.get_available_cameras()
        self.camera_selector.clear()

        if not cameras:
            self.camera_selector.addItem("未检测到摄像机", None)
        else:
            for cam in cameras:
                self.camera_selector.addItem(cam["name"], cam["id"])

        self.camera_thread = CameraThread(self.camera_manager, self.pose_processor)
        self.camera_thread.frames_ready.connect(self._on_live_frames)
        self.camera_thread.recording_progress.connect(self._on_recording_progress)
        self.camera_thread.recording_saved.connect(self._on_recording_saved)
        self.camera_thread.recording_too_short.connect(self._on_recording_too_short)
        self.camera_thread.start()

        if cameras:
            self.camera_manager.open_camera(cameras[0]["id"])

    def _on_camera_switched(self, index):
        cam_id = self.camera_selector.itemData(index)
        if cam_id is not None:
            self.camera_manager.open_camera(cam_id)

    # ------------------------------------------------------------------
    # 实时帧显示
    # ------------------------------------------------------------------
    def _on_live_frames(self, camera_bgr: np.ndarray, skeleton_bgr: np.ndarray):
        self._update_pixmap(self.camera_view, camera_bgr, attr="_last_camera_pixmap")
        self._update_pixmap(self.skeleton_view, skeleton_bgr, attr="_last_skeleton_pixmap")

    def _update_pixmap(self, label: QLabel, bgr: np.ndarray, attr: str):
        arr = np.ascontiguousarray(bgr)
        h, w, ch = arr.shape
        qimg = QImage(arr.data, w, h, ch * w, QImage.Format.Format_BGR888).copy()
        pixmap = QPixmap.fromImage(qimg)
        setattr(self, attr, pixmap)
        label.setPixmap(pixmap.scaled(
            label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    # ------------------------------------------------------------------
    # 录制控制
    # ------------------------------------------------------------------
    def _on_start_recording(self):
        if self.camera_thread is None:
            return
        os.makedirs(TEMP_DIR, exist_ok=True)
        uid = uuid.uuid4().hex[:8]
        video_path = os.path.join(TEMP_DIR, f"rec_{uid}.mp4")
        csv_path = os.path.join(TEMP_DIR, f"rec_{uid}.csv")

        self.camera_thread.start_recording(video_path, csv_path, fps=30.0)

        self.btn_start_rec.setEnabled(False)
        self.btn_stop_rec.setEnabled(True)
        self.camera_selector.setEnabled(False)
        self.lbl_rec_status.setText("录制中 0.0s …")

    def _on_stop_recording(self):
        if self.camera_thread is None:
            return
        self.btn_stop_rec.setEnabled(False)
        self.lbl_rec_status.setText("正在保存…")
        self.camera_thread.stop_recording()

    def _on_recording_progress(self, elapsed_sec: float, frames: float):
        self.lbl_rec_status.setText(f"录制中 {elapsed_sec:.1f}s  |  {int(frames)} 帧")

    def _on_recording_too_short(self):
        QMessageBox.warning(self, "录制过短",
                            "录制时间不足 10 秒，已丢弃。\n请重新录制。")
        self.btn_start_rec.setEnabled(True)
        self.btn_stop_rec.setEnabled(False)
        self.camera_selector.setEnabled(True)
        self.lbl_rec_status.setText("")

    def _on_recording_saved(self, video_path: str, csv_path: str,
                            total_frames: int, fps: float):
        self._switch_to_playback(video_path, csv_path, total_frames, fps)

    # ------------------------------------------------------------------
    # 模式切换
    # ------------------------------------------------------------------
    def _switch_to_playback(self, video_path: str, csv_path: str,
                             total_frames: int, fps: float):
        self._mode = "playback"
        self._playback_video_path = video_path
        self._playback_csv_path = csv_path
        self._playback_total_frames = total_frames
        self._playback_fps = fps

        # 隐藏录制控件，显示回放控件
        self.recording_bar.hide()
        self.playback_bar.show()

        self.slider_progress.setMaximum(max(0, total_frames - 1))
        self.slider_progress.setValue(0)
        self.lbl_frame_info.setText(f"0 / {total_frames}")

        # 停止实时摄像机线程
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        self.camera_manager.close_camera()

        # 启动回放线程
        self.playback_thread = PlaybackThread(video_path, csv_path)
        self.playback_thread.frames_ready.connect(self._on_playback_frames)
        self.playback_thread.playback_finished.connect(self._on_playback_finished)
        self.playback_thread.start()

        self.lbl_camera_title.setText("录制回放")
        self.lbl_skeleton_title.setText("骨架回放")

    # ------------------------------------------------------------------
    # 回放控制
    # ------------------------------------------------------------------
    def _on_playback_frames(self, camera_bgr: np.ndarray, skeleton_bgr: np.ndarray,
                            frame_idx: int):
        self._update_pixmap(self.camera_view, camera_bgr, attr="_last_camera_pixmap")
        self._update_pixmap(self.skeleton_view, skeleton_bgr, attr="_last_skeleton_pixmap")

        if not self._slider_dragging:
            self.slider_progress.setValue(frame_idx)

        self.lbl_frame_info.setText(
            f"{frame_idx} / {self._playback_total_frames}"
        )

    def _on_playback_finished(self):
        self.btn_play_pause.setText("▶ 播放")

    def _on_play_pause(self):
        if self.playback_thread is None:
            return
        if self.playback_thread.is_playing:
            self.playback_thread.pause()
            self.btn_play_pause.setText("▶ 播放")
        else:
            self.playback_thread.play()
            self.btn_play_pause.setText("⏸ 暂停")

    def _on_slider_pressed(self):
        self._slider_dragging = True

    def _on_slider_released(self):
        self._slider_dragging = False
        if self.playback_thread:
            self.playback_thread.seek(self.slider_progress.value())

    def _on_slider_moved(self, value: int):
        self.lbl_frame_info.setText(f"{value} / {self._playback_total_frames}")

    # ------------------------------------------------------------------
    # 提交分析 → 切换到分析页面
    # ------------------------------------------------------------------
    def _on_submit_analysis(self):
        if self.playback_thread:
            self.playback_thread.pause()

        # 清理旧的分析页面
        if self._analysis_page is not None:
            self._analysis_page.cleanup()
            self._stack.removeWidget(self._analysis_page)
            self._analysis_page.deleteLater()

        self._analysis_page = AnalysisPage(
            self._playback_video_path,
            self._playback_csv_path,
            parent=self,
        )
        self._analysis_page.back_requested.connect(self._on_back_from_analysis)
        self._stack.addWidget(self._analysis_page)
        self._stack.setCurrentWidget(self._analysis_page)
        self._analysis_page.start_analysis()

    def _on_back_from_analysis(self):
        if self._analysis_page is not None:
            self._analysis_page.cleanup()
            self._stack.removeWidget(self._analysis_page)
            self._analysis_page.deleteLater()
            self._analysis_page = None
        self._stack.setCurrentWidget(self._main_page)

    # ------------------------------------------------------------------
    # 事件
    # ------------------------------------------------------------------
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._last_camera_pixmap:
            self.camera_view.setPixmap(self._last_camera_pixmap.scaled(
                self.camera_view.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
        if self._last_skeleton_pixmap:
            self.skeleton_view.setPixmap(self._last_skeleton_pixmap.scaled(
                self.skeleton_view.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

    def closeEvent(self, event):
        if self.camera_thread:
            self.camera_thread.stop()
        if self.playback_thread:
            self.playback_thread.stop()
        if self._analysis_page:
            self._analysis_page.cleanup()
        self.camera_manager.close_camera()
        super().closeEvent(event)
