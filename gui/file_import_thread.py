import os
import logging
import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import QThread, Signal

from config import CSV_COLUMNS
from models.pose import PoseProcessor
from gui.frame_processor import FrameProcessor

logger = logging.getLogger(__name__)


class FileImportThread(QThread):
    """
    Perform frame-by-frame pose inference from an external video file, generating MP4 + CSV,
    then emit finished signal for the main window to switch to playback mode.
    """
    progress = Signal(int, int)       # (current_frame, total_frames)
    finished = Signal(str, str, int, float)  # (video_path, csv_path, total_frames, fps)
    error = Signal(str)

    def __init__(self, video_path: str, output_video: str, output_csv: str,
                 pose_processor: PoseProcessor):
        super().__init__()
        self.video_path = video_path
        self.output_video = output_video
        self.output_csv = output_csv
        self.pose_processor = pose_processor
        self._stop_requested = False
        self.skeleton_target_size = (1080, 1920)

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error.emit(f"Cannot open video file: {self.video_path}")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            if total_frames <= 0:
                cap.release()
                self.error.emit("Invalid video file (frame count is 0)")
                return

            os.makedirs(os.path.dirname(self.output_video) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            w, h = self.skeleton_target_size
            writer = cv2.VideoWriter(self.output_video, fourcc, fps, (w, h))

            csv_rows = []
            frame_id = 0

            while not self._stop_requested:
                ret, frame = cap.read()
                if not ret:
                    break

                processed = FrameProcessor.process(frame)
                pose_result = self.pose_processor.process(
                    processed, target_size=self.skeleton_target_size
                )

                cam_resized = cv2.resize(processed, (w, h), interpolation=cv2.INTER_LINEAR)
                writer.write(cam_resized)

                kpts_arr = pose_result.get("keypoints_array", {})
                xy = kpts_arr.get("xy")
                conf = kpts_arr.get("conf")
                if xy is not None and conf is not None:
                    row = [float(frame_id), 0.0]
                    for i in range(17):
                        row.extend([float(xy[i, 0]), float(xy[i, 1]), float(conf[i])])
                    csv_rows.append(row)

                frame_id += 1
                if frame_id % 10 == 0 or frame_id == total_frames:
                    self.progress.emit(frame_id, total_frames)

            cap.release()
            writer.release()

            if self._stop_requested:
                for path in (self.output_video, self.output_csv):
                    if os.path.exists(path):
                        os.remove(path)
                return

            df = pd.DataFrame(csv_rows, columns=CSV_COLUMNS)
            os.makedirs(os.path.dirname(self.output_csv) or ".", exist_ok=True)
            df.to_csv(self.output_csv, index=False)

            self.finished.emit(self.output_video, self.output_csv, frame_id, fps)

        except Exception as e:
            self.error.emit(str(e))
