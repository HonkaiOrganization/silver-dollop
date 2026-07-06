import logging
import cv2

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Camera hardware interface and resource management class.
    """
    def __init__(self):
        self.cap = None
        self.current_id = None

    def get_available_cameras(self) -> list:
        """
        Probe and return the list of available cameras on the current system.
        """
        available = []
        for i in range(3):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available.append({"id": i, "name": f"USB Camera {i}"})
                cap.release()
        return available

    def open_camera(self, camera_id: int):
        """Open camera by specified ID"""
        if self.cap is not None:
            self.close_camera()
        self.current_id = camera_id
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

    def read_frame(self):
        """Read current frame, returns None on failure"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def close_camera(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
