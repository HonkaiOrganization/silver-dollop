from __future__ import annotations

import logging
import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import QThread, Signal
from typing import Optional, cast

from config import KPT_NAMES
from models.pose import PoseProcessor

logger = logging.getLogger(__name__)


class PlaybackThread(QThread):
    """
    Replay camera frames and skeleton frames from MP4 + CSV.
    """
    frames_ready = Signal(object, object, int)  # (camera_bgr, skeleton_bgr, frame_idx)
    keypoints_ready = Signal(object, object)      # (xy [17,2], conf [17]) or (None, None)
    playback_finished = Signal()

    def __init__(self, video_path: str, csv_path: str, conf_thresh: float = 0.5):
        super().__init__()
        self.video_path = video_path
        self.csv_path = csv_path
        self.conf_thresh = conf_thresh
        self.skeleton_size = (1080, 1920)  # (w, h)

        self._is_running = False
        self._paused = False
        self._seek_to = -1
        self._playing = False

    # ------------------------------------------------------------------
    # Control interface
    # ------------------------------------------------------------------
    def play(self):
        self._paused = False
        self._playing = True

    def pause(self):
        self._paused = True

    def seek(self, frame_idx: int):
        self._seek_to = frame_idx

    def stop(self):
        self._is_running = False
        self.wait(2000)

    @property
    def is_playing(self) -> bool:
        return self._playing and not self._paused

    # ------------------------------------------------------------------
    # Thread main loop
    # ------------------------------------------------------------------
    def run(self):
        self._is_running = True

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Load CSV, pre-build index by frame_id (avoid filtering DataFrame per frame)
        df = pd.read_csv(self.csv_path)
        frame_kpts = self._build_frame_index(df, total_frames)

        frame_idx = 0
        frame_interval_ms = int(1000 / fps)

        while self._is_running:
            # Handle seek request
            if self._seek_to >= 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self._seek_to)
                frame_idx = self._seek_to
                self._seek_to = -1

            # Sleep when paused
            if self._paused or not self._playing:
                self.msleep(50)
                continue

            ret, camera_frame = cap.read()
            if not ret:
                self._paused = True
                self._playing = False
                self.playback_finished.emit()
                continue

            # Render skeleton from CSV
            skeleton_canvas = np.zeros((self.skeleton_size[1], self.skeleton_size[0], 3),
                                      dtype=np.uint8)
            if frame_idx < len(frame_kpts):
                xy, conf = frame_kpts[frame_idx]
                if xy is not None:
                    PoseProcessor.render_skeleton(skeleton_canvas, xy, conf, self.conf_thresh)
                self.keypoints_ready.emit(xy, conf)
            else:
                self.keypoints_ready.emit(None, None)

            self.frames_ready.emit(camera_frame, skeleton_canvas, frame_idx)
            frame_idx += 1
            self.msleep(frame_interval_ms)

        cap.release()

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------
    def _build_frame_index(self, df: pd.DataFrame, total_frames: int) -> list[tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
        """
        Build a list from DataFrame grouped by frame_id, each element is (xy, conf) or (None, None).
        xy: [17, 2] np.ndarray, conf: [17] np.ndarray
        """
        x_cols = [f"{n}_x" for n in KPT_NAMES]
        y_cols = [f"{n}_y" for n in KPT_NAMES]
        c_cols = [f"{n}_conf" for n in KPT_NAMES]

        index: list[tuple[Optional[np.ndarray], Optional[np.ndarray]]] = [
            (None, None) for _ in range(total_frames)
        ]

        if "frame_id" not in df.columns:
            return index

        grouped = df.groupby("frame_id")
        for fid, group in grouped:
            fid = int(cast(int, fid))
            if fid >= total_frames:
                continue

            if "person_id" in group.columns:
                person_ids = group["person_id"].unique()
                best_pid = person_ids[0]
                best_mean = -1.0
                for pid in person_ids:
                    m = group[group["person_id"] == pid][c_cols].mean().mean()
                    if m > best_mean:
                        best_mean = m
                        best_pid = pid
                row = group[group["person_id"] == best_pid].iloc[0]
            else:
                row = group.iloc[0]

            x_values = np.asarray(row[x_cols].tolist(), dtype=np.float64)
            y_values = np.asarray(row[y_cols].tolist(), dtype=np.float64)
            xy = np.stack((x_values, y_values), axis=1)  # [17, 2]
            conf = np.asarray(row[c_cols].tolist(), dtype=np.float64)  # [17]
            index[fid] = (xy, conf)

        return index
