# models/camera/__init__.py
import cv2

class CameraManager:
    """
    摄像机硬件调用与资源管理类。
    """
    def __init__(self):
        self.cap = None
        self.current_id = None

    def get_available_cameras(self) -> list:
        """
        探测并返回当前系统可用的摄像机列表。
        """
        available = []
        for i in range(3):
            # 使用 CAP_DSHOW 提高 Windows 下的稳定性，减少 MSMF 报错
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available.append({"id": i, "name": f"USB Camera {i}"})
                cap.release()
        return available

    def open_camera(self, camera_id: int):
        """打开指定ID的摄像机"""
        if self.cap is not None:
            self.close_camera()
        self.current_id = camera_id
        # 强制使用 DirectShow 后端
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

    def read_frame(self):
        """读取当前帧，若失败返回 None"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def close_camera(self):
        """释放摄像机资源"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None