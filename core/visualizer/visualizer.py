import logging
import cv2
import numpy as np
import pandas as pd

from config import KPT_NAMES, SKELETON_LINKS

logger = logging.getLogger(__name__)


def visualize_pose(csv_path: str, output_video_path: str, width: int = 1080, height: int = 1920, fps: float = 30.0):
    """
    Render CSV pose data as a skeleton video with pure black background.

    Args:
        csv_path: Input CSV file path
        output_video_path: Output video path
        width: Canvas width
        height: Canvas height
        fps: Output video frame rate

    Yields:
        float: Current rendering progress (0.0 ~ 1.0)
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV data is empty")

    frame_ids = sorted(df['frame_id'].unique())
    total_frames = len(frame_ids)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise ValueError(f"Cannot create video writer: {output_video_path}")

    try:
        for idx, fid in enumerate(frame_ids):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            frame_data = df[df['frame_id'] == fid]

            for _, row in frame_data.iterrows():
                xs = np.nan_to_num(row[[f"{n}_x" for n in KPT_NAMES]].values, nan=0.0)
                ys = np.nan_to_num(row[[f"{n}_y" for n in KPT_NAMES]].values, nan=0.0)
                confs = np.nan_to_num(row[[f"{n}_conf" for n in KPT_NAMES]].values, nan=0.0)

                for i, j in SKELETON_LINKS:
                    if confs[i] > 0.3 and confs[j] > 0.3:
                        pt1 = (int(xs[i]), int(ys[i]))
                        pt2 = (int(xs[j]), int(ys[j]))
                        cv2.line(canvas, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)

                for k in range(17):
                    if confs[k] > 0.3:
                        pt = (int(xs[k]), int(ys[k]))
                        cv2.circle(canvas, pt, 4, (0, 220, 255), -1, cv2.LINE_AA)

            writer.write(canvas)
            yield min((idx + 1) / total_frames, 1.0)

    finally:
        writer.release()
    yield 1.0
