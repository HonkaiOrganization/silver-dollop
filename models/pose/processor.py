import math
import logging
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import KPT_NAMES

logger = logging.getLogger(__name__)


class PoseProcessor:
    """
    姿态识别处理类。
    负责加载 YOLO 模型并提取人体骨架关键点。
    """
    KEYPOINT_NAMES = KPT_NAMES

    SKELETON_BONES = [
        ((15, 13), (0, 140, 255)),
        ((13, 11), (0, 165, 255)),
        ((16, 14), (255, 100, 0)),
        ((14, 12), (255, 140, 0)),
        ((11, 12), (255, 255, 255)),
        ((5, 6),   (255, 255, 255)),
        ((5, 11),  (200, 200, 200)),
        ((6, 12),  (200, 200, 200)),
        ((5, 7),   (0, 255, 0)),
        ((7, 9),   (0, 220, 100)),
        ((6, 8),   (255, 0, 255)),
        ((8, 10),  (200, 0, 255)),
        ((0, 1),   (0, 255, 255)), ((0, 2),   (0, 255, 255)),
        ((1, 3),   (0, 255, 255)), ((2, 4),   (0, 255, 255)),
        ((3, 5),   (0, 220, 255)), ((4, 6),   (0, 220, 255)),
    ]

    KPT_COLORS = [
        (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255),
        (0, 255, 100), (255, 100, 255), (0, 255, 0), (255, 0, 255), (0, 220, 0),
        (200, 0, 255), (0, 140, 255), (255, 140, 0), (0, 100, 255), (255, 100, 0),
        (0, 80, 255), (255, 80, 0),
    ]

    ANGLE_JOINTS = [
        ("L_Elb", 5,  7,  9,  (0, 255, 0)),
        ("R_Elb", 6,  8,  10, (255, 0, 255)),
        ("L_Kne", 11, 13, 15, (0, 140, 255)),
        ("R_Kne", 12, 14, 16, (255, 140, 0)),
        ("L_Shoulder", 7,  5,  11, (0, 255, 100)),
        ("R_Shoulder", 8,  6,  12, (255, 100, 255)),
        ("L_Hip", 5,  11, 13, (0, 165, 255)),
        ("R_Hip", 6,  12, 14, (255, 165, 0)),
    ]

    _BONE_THICKNESS = 4
    _KPT_OUTER_RADIUS = 14
    _KPT_INNER_RADIUS = 10
    _ARC_RADIUS = 44
    _FONT = cv2.FONT_HERSHEY_SIMPLEX
    _FONT_SCALE = 0.62
    _FONT_THICKNESS = 2

    def __init__(
        self,
        model_path: str = 'pretrained/yolo11m-pose.pt',
        conf_thresh: float = 0.5,
        max_persons: int = 1,
        device: str | None = None,
    ):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.max_persons = max_persons
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = YOLO(self.model_path)
            self._model.to(self.device)
        return self._model

    @staticmethod
    def _calc_angle_deg(a: tuple, b: tuple, c: tuple) -> float:
        """计算向量 ba 与 bc 在顶点 b 处的夹角（度）"""
        ba = np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64)
        bc = np.array(c, dtype=np.float64) - np.array(b, dtype=np.float64)
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    @staticmethod
    def _draw_angle_arc(canvas, vertex, pt_a, pt_c, angle_deg, color,
                        arc_radius=44, font=cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale=0.62, font_thickness=2):
        """在顶点处绘制角度弧线及数值标注"""
        vx, vy = vertex
        ang_a = math.degrees(math.atan2(pt_a[1] - vy, pt_a[0] - vx))
        ang_c = math.degrees(math.atan2(pt_c[1] - vy, pt_c[0] - vx))

        start_ang = min(ang_a, ang_c)
        end_ang = max(ang_a, ang_c)
        if end_ang - start_ang > 180:
            start_ang, end_ang = end_ang, start_ang + 360

        overlay = canvas.copy()
        cv2.ellipse(overlay, vertex, (arc_radius, arc_radius),
                    0, start_ang, end_ang, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

        cv2.ellipse(canvas, vertex, (arc_radius, arc_radius),
                    0, start_ang, end_ang, color, 2, cv2.LINE_AA)

        mid_ang = math.radians((start_ang + end_ang) / 2)
        text_dist = arc_radius + 22
        tx = int(vx + text_dist * math.cos(mid_ang))
        ty = int(vy + text_dist * math.sin(mid_ang))

        label = f"{angle_deg:.0f}deg"
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(canvas,
                      (tx - 4, ty - th - 4),
                      (tx + tw + 4, ty + baseline + 4),
                      (0, 0, 0), -1)
        cv2.putText(canvas, label, (tx, ty), font, font_scale, color,
                    font_thickness, cv2.LINE_AA)

    @classmethod
    def render_skeleton(cls, canvas, xy, conf, conf_thresh=0.5):
        """
        在画布上绘制骨架（静态方法，可供录制回放等场景复用）。
        """
        valid_kpts = []
        for i in range(17):
            if conf[i] > conf_thresh:
                valid_kpts.append((int(xy[i, 0]), int(xy[i, 1])))
            else:
                valid_kpts.append(None)

        for (idx1, idx2), color in cls.SKELETON_BONES:
            pt1, pt2 = valid_kpts[idx1], valid_kpts[idx2]
            if pt1 and pt2:
                cv2.line(canvas, (pt1[0]+2, pt1[1]+2), (pt2[0]+2, pt2[1]+2),
                         (30, 30, 30), cls._BONE_THICKNESS+1, cv2.LINE_AA)
                cv2.line(canvas, pt1, pt2, color, cls._BONE_THICKNESS, cv2.LINE_AA)

        for i, pt in enumerate(valid_kpts):
            if pt:
                cv2.circle(canvas, pt, cls._KPT_OUTER_RADIUS, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, pt, cls._KPT_INNER_RADIUS, cls.KPT_COLORS[i], -1, cv2.LINE_AA)

        for label, idx_a, idx_v, idx_c, color in cls.ANGLE_JOINTS:
            pt_a, pt_v, pt_c = valid_kpts[idx_a], valid_kpts[idx_v], valid_kpts[idx_c]
            if pt_a and pt_v and pt_c:
                angle = cls._calc_angle_deg(pt_a, pt_v, pt_c)
                cls._draw_angle_arc(canvas, pt_v, pt_a, pt_c, angle, color,
                                    arc_radius=cls._ARC_RADIUS, font=cls._FONT,
                                    font_scale=cls._FONT_SCALE,
                                    font_thickness=cls._FONT_THICKNESS)

    def process(self, frame: np.ndarray, target_size: tuple = (1080, 1920)) -> dict:
        """
        执行姿态推理，返回结构化数据。
        """
        h, w = frame.shape[:2]
        input_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

        imgsz = ((target_size[0] + 31) // 32) * 32

        results = self.model(
            input_frame,
            verbose=False,
            imgsz=imgsz,
            device=self.device,
        )
        result = results[0]

        skeleton_canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        keypoints_data = []
        best_xy, best_conf = None, None

        if result.keypoints is not None and result.keypoints.data.shape[0] > 0:
            xy = result.keypoints.xy.cpu().numpy()
            conf = result.keypoints.conf.cpu().numpy()

            num_persons = xy.shape[0]
            mean_confs = np.array([conf[pid].mean() for pid in range(num_persons)])
            ranked_ids = np.argsort(-mean_confs)[:self.max_persons]

            if len(ranked_ids) > 0:
                best_xy = xy[ranked_ids[0]]
                best_conf = conf[ranked_ids[0]]

            for person_id in ranked_ids:
                person_kpts = []
                for i in range(17):
                    if conf[person_id, i] > self.conf_thresh:
                        person_kpts.append({
                            "name": self.KEYPOINT_NAMES[i],
                            "x": float(xy[person_id, i, 0]),
                            "y": float(xy[person_id, i, 1]),
                            "conf": float(conf[person_id, i])
                        })
                    else:
                        person_kpts.append(None)

                keypoints_data.append({
                    "person_id": int(person_id),
                    "keypoints": person_kpts
                })

        if best_xy is not None and best_conf is not None:
            self.render_skeleton(skeleton_canvas, best_xy, best_conf, self.conf_thresh)

        return {
            "skeleton_image": skeleton_canvas,
            "keypoints": keypoints_data,
            "keypoints_array": {"xy": best_xy, "conf": best_conf},
            "source_resolution": (w, h),
            "target_resolution": target_size
        }
