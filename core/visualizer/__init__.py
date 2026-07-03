import cv2
import numpy as np
import pandas as pd

# COCO 17点骨架连接定义
SKELETON_LINKS = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [7, 9], 
    [6, 8], [8, 10], [0, 1], [1, 3], [0, 2], [2, 4]
]

KPT_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_sho", "R_sho", "L_elb", "R_elb",
    "L_wri", "R_wri", "L_hip", "R_hip",
    "L_kne", "R_kne", "L_ank", "R_ank"
]

def visualize_pose(csv_path: str, output_video_path: str, width: int = 1080, height: int = 1920, fps: float = 30.0):
    """
    将CSV姿态数据渲染为纯黑背景的骨架视频。
    
    Args:
        csv_path: 输入CSV文件路径
        output_video_path: 输出视频路径
        width: 画布宽度
        height: 画布高度
        fps: 输出视频帧率
        
    Yields:
        float: 当前渲染进度 (0.0 ~ 1.0)
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV数据为空")
        
    frame_ids = sorted(df['frame_id'].unique())
    total_frames = len(frame_ids)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not writer.isOpened():
        raise ValueError(f"无法创建视频写入器: {output_video_path}")
        
    try:
        for idx, fid in enumerate(frame_ids):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            frame_data = df[df['frame_id'] == fid]
            
            for _, row in frame_data.iterrows():
                xs = np.nan_to_num(row[[f"{n}_x" for n in KPT_NAMES]].values, nan=0.0)
                ys = np.nan_to_num(row[[f"{n}_y" for n in KPT_NAMES]].values, nan=0.0)
                confs = np.nan_to_num(row[[f"{n}_conf" for n in KPT_NAMES]].values, nan=0.0)
                
                for link in SKELETON_LINKS:
                    i, j = link
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