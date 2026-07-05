import os
import sys
import math
import uuid
import webbrowser
import logging
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QComboBox, QSizePolicy,
    QPushButton, QSlider, QMessageBox, QStackedWidget,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

from models.camera import CameraManager
from models.pose import PoseProcessor
from gui.camera_thread import CameraThread
from gui.playback_thread import PlaybackThread
from gui.analysis_page import AnalysisPage
from gui.file_import_thread import FileImportThread
from gui.workers.realtime_inference_thread import RealtimeInferenceThread
from config import KPT_NAMES

logger = logging.getLogger(__name__)

TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.camera_manager = CameraManager()
        self.pose_processor = PoseProcessor()
        self.camera_thread: CameraThread | None = None
        self.playback_thread: PlaybackThread | None = None

        self._last_camera_pixmap = None
        self._last_skeleton_pixmap = None
        self._mode = "recording"
        self._import_thread: FileImportThread | None = None
        self._import_dialog: QWidget | None = None

        self._countdown_timer: QTimer | None = None
        self._countdown_value = 0
        self._pending_video_path = ""
        self._pending_csv_path = ""
        self._inference_thread: RealtimeInferenceThread | None = None

        self.init_ui()
        self.init_inference_thread()
        self.init_camera_system()

    def init_ui(self):
        self.setWindowTitle("跳绳姿态录制与分析")

        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        width = int(screen_geometry.width() * 0.8)
        height = int(width * 9 / 16)
        self.resize(width, height)

        file_menu = self.menuBar().addMenu("文件(&F)")
        act_open = file_menu.addAction("打开视频文件(&O)…")
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._on_open_video_file)
        file_menu.addSeparator()
        act_exit = file_menu.addAction("退出(&X)")
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)

        help_menu = self.menuBar().addMenu("帮助(&H)")
        act_help = help_menu.addAction("查看帮助文档(&H)")
        act_help.setShortcut("F1")
        act_help.triggered.connect(self._on_show_help)
        act_about = help_menu.addAction("关于(&A)")
        act_about.triggered.connect(self._on_show_about)

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._main_page = QWidget()
        main_layout = QVBoxLayout(self._main_page)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)
        self._stack.addWidget(self._main_page)

        self._analysis_page: AnalysisPage | None = None

        self.recording_bar = QWidget()
        rec_layout = QHBoxLayout(self.recording_bar)
        rec_layout.setContentsMargins(0, 0, 0, 0)

        self.camera_selector = QComboBox()
        self.camera_selector.setMinimumWidth(150)
        self.camera_selector.currentIndexChanged.connect(self._on_camera_switched)

        self.btn_start_rec = QPushButton("● 开始记录")
        self.btn_start_rec.clicked.connect(self._on_start_recording)

        self.btn_stop_rec = QPushButton("■ 结束记录")
        self.btn_stop_rec.setEnabled(False)
        self.btn_stop_rec.clicked.connect(self._on_stop_recording)

        self.lbl_rec_status = QLabel("")

        rec_layout.addWidget(QLabel("摄像机:"))
        rec_layout.addWidget(self.camera_selector)
        rec_layout.addSpacing(20)
        rec_layout.addWidget(self.btn_start_rec)
        rec_layout.addWidget(self.btn_stop_rec)
        rec_layout.addWidget(self.lbl_rec_status)
        rec_layout.addStretch()

        self.playback_bar = QWidget()
        pb_layout = QHBoxLayout(self.playback_bar)
        pb_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_play_pause = QPushButton("▶ 播放")
        self.btn_play_pause.setFixedWidth(90)
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

        self.btn_rerecord = QPushButton("⟳ 重新录制")
        self.btn_rerecord.clicked.connect(self._on_rerecord)

        self.btn_submit = QPushButton("提交分析")
        self.btn_submit.clicked.connect(self._on_submit_analysis)

        pb_layout.addWidget(self.btn_play_pause)
        pb_layout.addSpacing(10)
        pb_layout.addWidget(self.slider_progress, 1)
        pb_layout.addWidget(self.lbl_frame_info)
        pb_layout.addSpacing(12)
        pb_layout.addWidget(self.btn_rerecord)
        pb_layout.addSpacing(8)
        pb_layout.addWidget(self.btn_submit)

        self.playback_bar.hide()

        views_layout = QHBoxLayout()
        views_layout.setSpacing(10)

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.camera_view = QLabel()
        self.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_view.setStyleSheet("background-color: black;")
        self.camera_view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.camera_view.setMinimumSize(1, 1)

        self.lbl_camera_title = QLabel("Camera Feed")
        self.lbl_camera_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        left_layout.addWidget(self.lbl_camera_title)
        left_layout.addWidget(self.camera_view, 1)

        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.skeleton_view = QLabel()
        self.skeleton_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.skeleton_view.setStyleSheet("background-color: black;")
        self.skeleton_view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.skeleton_view.setMinimumSize(1, 1)

        self.lbl_skeleton_title = QLabel("Pose Skeleton")
        self.lbl_skeleton_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        right_layout.addWidget(self.lbl_skeleton_title)
        right_layout.addWidget(self.skeleton_view, 1)

        views_layout.addWidget(left_container, 1)
        views_layout.addWidget(right_container, 1)

        # ── 关键点坐标表格 + 置信度 ─────────────────────────────────
        kpt_panel = QWidget()
        kpt_panel.setFixedWidth(280)
        kpt_panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Ignored)
        kpt_panel_layout = QVBoxLayout(kpt_panel)
        kpt_panel_layout.setContentsMargins(0, 0, 0, 0)
        kpt_panel_layout.setSpacing(4)

        self.kpt_table = QTableWidget(17, 3)
        self.kpt_table.setHorizontalHeaderLabels(["关键点", "X", "Y"])
        self.kpt_table.verticalHeader().setVisible(False)
        self.kpt_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.kpt_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.kpt_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.kpt_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.kpt_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for i, name in enumerate(KPT_NAMES):
            item = QTableWidgetItem(name)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.kpt_table.setItem(i, 0, item)
            self.kpt_table.setItem(i, 1, QTableWidgetItem("-"))
            self.kpt_table.setItem(i, 2, QTableWidgetItem("-"))
            for col in range(1, 3):
                self.kpt_table.item(i, col).setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        kpt_panel_layout.addWidget(self.kpt_table, 1)

        self.lbl_confidence = QLabel("Abnormal: NaN")
        self.lbl_confidence.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_confidence.setStyleSheet(
            "font-size: 15px; font-weight: bold;"
            "background: rgba(0,0,0,180); color: white;"
            "padding: 5px 10px; border-radius: 4px;"
        )
        kpt_panel_layout.addWidget(self.lbl_confidence)

        views_layout.addWidget(kpt_panel)

        self.lbl_countdown = QLabel("")
        self.lbl_countdown.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_countdown.setStyleSheet(
            "font-size: 120px; font-weight: bold;"
            "background: rgba(0,0,0,150); color: white;"
        )
        self.lbl_countdown.setFixedSize(200, 200)
        self.lbl_countdown.hide()

        views_container = QWidget()
        views_stack = QVBoxLayout(views_container)
        views_stack.setContentsMargins(0, 0, 0, 0)
        views_stack.addLayout(views_layout, 1)

        main_layout.addWidget(self.recording_bar)
        main_layout.addWidget(self.playback_bar)
        main_layout.addWidget(views_container, 1)

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
        self.camera_thread.keypoints_ready.connect(self._on_keypoints_updated)
        self.camera_thread.keypoints_ready.connect(self._inference_thread.add_keypoints)
        self.camera_thread.recording_progress.connect(self._on_recording_progress)
        self.camera_thread.recording_saved.connect(self._on_recording_saved)
        self.camera_thread.recording_too_short.connect(self._on_recording_too_short)
        self.camera_thread.start()

        if cameras:
            self.camera_manager.open_camera(cameras[0]["id"])

    def init_inference_thread(self):
        self._inference_thread = RealtimeInferenceThread()
        self._inference_thread.result_ready.connect(self._on_inference_result)
        self._inference_thread.start()

    def _on_camera_switched(self, index):
        cam_id = self.camera_selector.itemData(index)
        if cam_id is not None:
            self.camera_manager.open_camera(cam_id)

    def _on_live_frames(self, camera_bgr: np.ndarray, skeleton_bgr: np.ndarray):
        self._update_pixmap(self.camera_view, camera_bgr, attr="_last_camera_pixmap")
        self._update_pixmap(self.skeleton_view, skeleton_bgr, attr="_last_skeleton_pixmap")

    def _on_keypoints_updated(self, xy, conf):
        for i in range(17):
            if xy is not None:
                x_val = f"{xy[i, 0]:.1f}"
                y_val = f"{xy[i, 1]:.1f}"
            else:
                x_val = "-"
                y_val = "-"
            self.kpt_table.item(i, 1).setText(x_val)
            self.kpt_table.item(i, 2).setText(y_val)

    def _on_inference_result(self, confidence: float):
        if math.isnan(confidence):
            self.lbl_confidence.setText("Abnormal: NaN")
        else:
            self.lbl_confidence.setText(f"Abnormal: {confidence:.2%}")

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

    def _on_start_recording(self):
        if self.camera_thread is None:
            return
        os.makedirs(TEMP_DIR, exist_ok=True)
        uid = uuid.uuid4().hex[:8]
        self._pending_video_path = os.path.join(TEMP_DIR, f"rec_{uid}.mp4")
        self._pending_csv_path = os.path.join(TEMP_DIR, f"rec_{uid}.csv")

        self.btn_start_rec.setEnabled(False)
        self.btn_stop_rec.setEnabled(False)
        self.camera_selector.setEnabled(False)
        self.lbl_rec_status.setText("准备录制…")

        self._countdown_value = 3
        self.lbl_countdown.setText("3")
        self.lbl_countdown.show()

        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._on_countdown_tick)
        self._countdown_timer.start()

    def _on_countdown_tick(self):
        self._countdown_value -= 1
        if self._countdown_value > 0:
            self.lbl_countdown.setText(str(self._countdown_value))
            self.lbl_rec_status.setText(f"准备录制… {self._countdown_value}")
        else:
            if self._countdown_timer is None:
                return
            self._countdown_timer.stop()
            self._countdown_timer = None
            self.lbl_countdown.hide()

            if self.camera_thread is None:
                return
            self.camera_thread.start_recording(
                self._pending_video_path, self._pending_csv_path, fps=30.0
            )
            self.btn_stop_rec.setEnabled(True)
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

    def _switch_to_playback(self, video_path: str, csv_path: str,
                             total_frames: int, fps: float):
        self._mode = "playback"
        self._playback_video_path = video_path
        self._playback_csv_path = csv_path
        self._playback_total_frames = total_frames
        self._playback_fps = fps

        self.recording_bar.hide()
        self.playback_bar.show()

        self.slider_progress.setMaximum(max(0, total_frames - 1))
        self.slider_progress.setValue(0)
        self.lbl_frame_info.setText(f"0 / {total_frames}")

        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        self.camera_manager.close_camera()

        self.playback_thread = PlaybackThread(video_path, csv_path)
        self.playback_thread.frames_ready.connect(self._on_playback_frames)
        self.playback_thread.keypoints_ready.connect(self._on_keypoints_updated)
        self.playback_thread.keypoints_ready.connect(self._inference_thread.add_keypoints)
        self.playback_thread.playback_finished.connect(self._on_playback_finished)
        self.playback_thread.start()

        self._inference_thread.reset()
        self.lbl_confidence.setText("Abnormal: NaN")

        self.lbl_camera_title.setText("录制回放")
        self.lbl_skeleton_title.setText("骨架回放")

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

    def _on_rerecord(self):
        if self.playback_thread:
            self.playback_thread.stop()
            self.playback_thread = None

        self.camera_view.clear()
        self.skeleton_view.clear()
        self._last_camera_pixmap = None
        self._last_skeleton_pixmap = None

        for i in range(17):
            self.kpt_table.item(i, 1).setText("-")
            self.kpt_table.item(i, 2).setText("-")

        self._inference_thread.reset()
        self.lbl_confidence.setText("Abnormal: NaN")

        self._mode = "recording"
        self.playback_bar.hide()
        self.recording_bar.show()

        self.btn_start_rec.setEnabled(True)
        self.btn_stop_rec.setEnabled(False)
        self.camera_selector.setEnabled(True)
        self.lbl_rec_status.setText("")

        self.lbl_camera_title.setText("Camera Feed")
        self.lbl_skeleton_title.setText("Pose Skeleton")

        self.init_camera_system()

    def _on_submit_analysis(self):
        if self.playback_thread:
            self.playback_thread.pause()

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

    def _on_open_video_file(self):
        if self._import_thread and self._import_thread.isRunning():
            QMessageBox.information(self, "提示", "正在导入视频，请稍候…")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开视频文件", "",
            "视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv);;所有文件 (*)"
        )
        if not file_path:
            return

        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        self.camera_manager.close_camera()

        uid = uuid.uuid4().hex[:8]
        video_out = os.path.join(TEMP_DIR, f"imp_{uid}.mp4")
        csv_out = os.path.join(TEMP_DIR, f"imp_{uid}.csv")

        self._import_dialog = QWidget(None, Qt.WindowType.Window
                                      | Qt.WindowType.WindowStaysOnTopHint
                                      | Qt.WindowType.CustomizeWindowHint
                                      | Qt.WindowType.WindowTitleHint
                                      | Qt.WindowType.WindowCloseButtonHint)
        self._import_dialog.setWindowTitle("导入视频")
        self._import_dialog.setFixedSize(280, 80)
        dlg_layout = QVBoxLayout(self._import_dialog)
        dlg_layout.setContentsMargins(20, 12, 20, 12)
        lbl = QLabel("正在推理 CSV，请稍候…")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dlg_layout.addWidget(lbl)
        self._import_dialog.closeEvent = self._on_import_dialog_close
        geo = self.geometry()
        dw, dh = 280, 80
        self._import_dialog.move(
            geo.x() + (geo.width() - dw) // 2,
            geo.y() + (geo.height() - dh) // 2,
        )
        self._import_dialog.show()

        self._import_thread = FileImportThread(
            file_path, video_out, csv_out, self.pose_processor
        )
        self._import_thread.finished.connect(self._on_import_finished)
        self._import_thread.error.connect(self._on_import_error)
        self._import_thread.start()

    def _on_import_dialog_close(self, event):
        if self._import_thread and self._import_thread.isRunning():
            self._import_thread.stop()
        self._import_thread = None
        self._import_dialog = None
        self.init_camera_system()
        event.accept()

    def _on_import_finished(self, video_path: str, csv_path: str,
                             total_frames: int, fps: float):
        dlg = self._import_dialog
        self._import_dialog = None
        if dlg:
            dlg.close()
        self._import_thread = None
        self._switch_to_playback(video_path, csv_path, total_frames, fps)

    def _on_import_error(self, msg: str):
        dlg = self._import_dialog
        self._import_dialog = None
        if dlg:
            dlg.close()
        self._import_thread = None
        QMessageBox.critical(self, "导入失败", f"视频导入出错:\n{msg}")
        self.init_camera_system()

    def _on_show_help(self):
        readme_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "README.md"
        )
        if os.path.exists(readme_path):
            webbrowser.open(readme_path)
        else:
            QMessageBox.information(
                self, "帮助文档",
                f"README.md 不存在于:\n{readme_path}"
            )

    def _on_show_about(self):
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox
        from PySide6.QtCore import Qt
        import PySide6
        import torch
        import cv2
        import ultralytics

        versions = [
            ("Python", sys.version.split()[0]),
            ("PySide6", PySide6.__version__),
            ("PyTorch", torch.__version__),
            ("OpenCV", cv2.__version__),
            ("Ultralytics", ultralytics.__version__),
        ]

        dialog = QDialog(self)
        dialog.setWindowTitle("关于")
        dialog.setFixedSize(300, 240)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(8)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("跳绳姿态录制与分析  v1.0.0")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        for name, ver in versions:
            lbl = QLabel(f"{name}:  {ver}")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-size: 12px;")
            layout.addWidget(lbl)

        layout.addStretch()

        link = QLabel(
            "<a href='https://github.com/HonkaiOrganization'>Honkai Organization</a>"
        )
        link.setAlignment(Qt.AlignmentFlag.AlignCenter)
        link.setOpenExternalLinks(True)
        layout.addWidget(link)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)

        dialog.exec()

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
        if self._countdown_timer:
            self._countdown_timer.stop()
        if self.camera_thread:
            self.camera_thread.stop()
        if self.playback_thread:
            self.playback_thread.stop()
        if self._inference_thread:
            self._inference_thread.stop()
        if self._import_thread and self._import_thread.isRunning():
            self._import_thread.stop()
        if self._import_dialog:
            self._import_dialog.close()
        if self._analysis_page:
            self._analysis_page.cleanup()
        self.camera_manager.close_camera()
        super().closeEvent(event)
