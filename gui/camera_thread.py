import os
import time
import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import QThread, Signal

from models.camera import CameraManager
from models.pose import PoseProcessor
from gui.frame_processor import FrameProcessor


# 与 PoseExtractor 保持一致的 CSV 列定义
KPT_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_sho", "R_sho", "L_elb", "R_elb",
    "L_wri", "R_wri", "L_hip", "R_hip",
    "L_kne", "R_kne", "L_ank", "R_ank"
]
CSV_COLUMNS = ["frame_id", "person_id"] + [
    f"{name}_{suffix}" for name in KPT_NAMES for suffix in ("x", "y", "conf")
]


class CameraThread(QThread):
    """
    摄像机读取与姿态推理联合线程，支持录制。
    """
    frames_ready = Signal(object, object)          # (camera_bgr, skeleton_bgr)
    recording_progress = Signal(float, float)      # (elapsed_sec, frames_written)
    recording_saved = Signal(str, str, int, float) # (video_path, csv_path, total_frames, fps)
    recording_too_short = Signal()                 # 录制不足 10 秒

    def __init__(self, camera_manager: CameraManager, pose_processor: PoseProcessor):
        super().__init__()
        self.camera_manager = camera_manager
        self.pose_processor = pose_processor
        self._is_running = False
        self.skeleton_target_size = (1080, 1920)

        self._recording = False
        self._record_start_time = 0.0
        self._record_frame_id = 0
        self._video_writer = None
        self._csv_rows = []
        self._video_path = ""
        self._csv_path = ""
        self._record_fps = 0.0
        self._stop_requested = False

    def start_recording(self, video_path: str, csv_path: str, fps: float = 30.0):
        """开始录制（在下一帧处理时生效）"""
        self._video_path = video_path
        self._csv_path = csv_path
        self._record_fps = fps
        self._record_frame_id = 0
        self._csv_rows = []
        self._record_start_time = time.time()
        self._recording = True
        self._stop_requested = False

        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        w, h = self.skeleton_target_size
        self._video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    def stop_recording(self):
        """请求停止录制（实际保存在线程内完成，通过信号通知主线程）"""
        self._stop_requested = True

    def is_recording(self) -> bool:
        return self._recording

    def run(self):
        self._is_running = True
        while self._is_running:
            frame = self.camera_manager.read_frame()
            if frame is None:
                self.msleep(30)
                continue

            processed_camera = FrameProcessor.process(frame)
            pose_result = self.pose_processor.process(
                processed_camera, target_size=self.skeleton_target_size
            )

            self.frames_ready.emit(processed_camera, pose_result["skeleton_image"])

            if self._recording:
                self._record_frame(processed_camera, pose_result)

            if self._stop_requested and self._recording:
                self._recording = False
                self._finish_recording()

    def stop(self):
        self._is_running = False
        self.wait(3000)

    def _record_frame(self, processed_camera: np.ndarray, pose_result: dict):
        """将当前帧写入录制文件"""
        w, h = self.skeleton_target_size
        cam_resized = cv2.resize(processed_camera, (w, h), interpolation=cv2.INTER_LINEAR)
        if self._video_writer is not None:
            self._video_writer.write(cam_resized)

        kpts_arr = pose_result.get("keypoints_array", {})
        xy = kpts_arr.get("xy")
        conf = kpts_arr.get("conf")
        if xy is not None and conf is not None:
            row = [float(self._record_frame_id), 0.0]
            for i in range(17):
                row.extend([float(xy[i, 0]), float(xy[i, 1]), float(conf[i])])
            self._csv_rows.append(row)

        self._record_frame_id += 1
        elapsed = time.time() - self._record_start_time
        self.recording_progress.emit(elapsed, float(self._record_frame_id))

    def _finish_recording(self):
        """关闭文件并发送录制完成信号"""
        print(f"[CameraThread] 录制完成，保存视频: {self._video_path}, CSV: {self._csv_path}")
        elapsed = time.time() - self._record_start_time

        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

        if elapsed < 10.0:
            for path in (self._video_path, self._csv_path):
                if path and os.path.exists(path):
                    os.remove(path)
            self.recording_too_short.emit()
            return

        df = pd.DataFrame(self._csv_rows, columns=CSV_COLUMNS)
        os.makedirs(os.path.dirname(self._csv_path) or ".", exist_ok=True)
        df.to_csv(self._csv_path, index=False)
        self._csv_rows = []

        self.recording_saved.emit(
            self._video_path, self._csv_path,
            self._record_frame_id, self._record_fps
        )
