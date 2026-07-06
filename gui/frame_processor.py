import numpy as np


class FrameProcessor:
    """
    Base image frame processing class.
    Passes through 9:16 aspect ratio unchanged; all other ratios are center-cropped.
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
