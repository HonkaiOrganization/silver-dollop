import sys
from PySide6.QtWidgets import QApplication, QMainWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("16:9 GUI Window")
        
        # 获取主屏幕对象及其可用几何尺寸
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        
        # 设定窗口宽度为屏幕可用宽度的 80%
        width = int(screen_geometry.width() * 0.8)
        # 根据 16:9 比例计算高度 (高度 = 宽度 * 9 / 16)
        height = int(width * 9 / 16)
        
        # 应用计算后的尺寸
        self.resize(width, height)
        
        # 如需固定尺寸，可直接使用 self.resize(1280, 720)
