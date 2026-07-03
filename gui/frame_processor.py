import numpy as np


class FrameProcessor:
    """
    基础图像帧处理类。
    仅保留 9:16 比例直通，其余比例一律执行中心裁剪。
    """
    TARGET_ASPECT = 9 / 16
    TOLERANCE = 0.05

    @classmethod
    def process(cls, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            return None

        h, w = frame.shape[:2]
        current_aspect = w / h

        if abs(current_aspect - cls.TARGET_ASPECT) < cls.TOLERANCE:
            return frame

        if current_aspect > cls.TARGET_ASPECT:
            new_w = int(h * cls.TARGET_ASPECT)
            start_x = (w - new_w) // 2
            return frame[:, start_x:start_x + new_w]
        else:
            new_h = int(w / cls.TARGET_ASPECT)
            start_y = (h - new_h) // 2
            return frame[start_y:start_y + new_h, :]
