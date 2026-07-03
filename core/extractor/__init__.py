import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

class PoseExtractor:
    def __init__(self, model_path: str = 'pretrained/yolo11n-pose.pt', imgsz: int = 640):
        self.model_path = model_path
        self.imgsz = imgsz
        self.model = YOLO(model_path)

    def extract_pose(self, video_path: str, output_csv_path: str, target_size: tuple = (1080, 1920)):
        """
        提取视频姿态数据并保存为CSV。
        
        Args:
            video_path: 输入视频路径
            output_csv_path: 输出CSV文件路径
            target_size: 目标分辨率 (width, height)，默认 (1080, 1920)
            
        Yields:
            float: 当前处理进度 (0.0 ~ 1.0)
        """
        kpt_names = [
            "nose", "L_eye", "R_eye", "L_ear", "R_ear",
            "L_sho", "R_sho", "L_elb", "R_elb",
            "L_wri", "R_wri", "L_hip", "R_hip",
            "L_kne", "R_kne", "L_ank", "R_ank"
        ]
        
        columns = ["frame_id", "person_id"]
        for name in kpt_names:
            columns.extend([f"{name}_x", f"{name}_y", f"{name}_conf"])
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError("视频帧数无效")
        
        all_data = []
        frame_id = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 将视频帧缩放至目标分辨率
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
            
        df = pd.DataFrame(all_data, columns=columns) if all_data else pd.DataFrame(columns=columns)
        os.makedirs(os.path.dirname(output_csv_path) or '.', exist_ok=True)
        df.to_csv(output_csv_path, index=False)