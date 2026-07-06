import os
import logging
import cv2
import pandas as pd
from ultralytics import YOLO

from config import KPT_NAMES, CSV_COLUMNS

logger = logging.getLogger(__name__)


class PoseExtractor:
    def __init__(self, model_path: str = 'pretrained/yolo11n-pose.pt', imgsz: int = 640):
        self.model_path = model_path
        self.imgsz = imgsz
        self.model = YOLO(model_path)

    def extract_pose(self, video_path: str, output_csv_path: str, target_size: tuple = (1080, 1920)):
        """
        Extract pose data from video and save as CSV.

        Args:
            video_path: Input video path
            output_csv_path: Output CSV file path
            target_size: Target resolution (width, height), default (1080, 1920)

        Yields:
            float: Current processing progress (0.0 ~ 1.0)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError("Invalid video frame count")

        all_data = []
        frame_id = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

                results = self.model(frame, verbose=False, imgsz=self.imgsz)
                result = results[0]

                if result.keypoints is not None and result.keypoints.data.shape[0] > 0:
                    keypoints_xy = result.keypoints.xy.cpu().numpy()
                    keypoints_conf = result.keypoints.conf.cpu().numpy()

                    for person_id in range(keypoints_xy.shape[0]):
                        row = [float(frame_id), float(person_id)]
                        for i in range(17):
                            row.extend([
                                float(keypoints_xy[person_id, i, 0]),
                                float(keypoints_xy[person_id, i, 1]),
                                float(keypoints_conf[person_id, i])
                            ])
                        all_data.append(row)

                frame_id += 1
                yield min(frame_id / total_frames, 1.0)

        finally:
            cap.release()

        df = pd.DataFrame(all_data, columns=CSV_COLUMNS) if all_data else pd.DataFrame(columns=CSV_COLUMNS)
        os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
        df.to_csv(output_csv_path, index=False)
