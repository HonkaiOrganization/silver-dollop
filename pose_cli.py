import os
import sys
import glob
import json
import cv2
import math
import numpy as np
import pandas as pd
import click

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QSlider, QLabel, QSizePolicy)
from PySide6.QtCore import Qt, QTimer, QRectF, QRect, QPointF, QThread, PySideSignal
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QFontMetrics, QBrush
from ultralytics import YOLO

# ================= 辅助函数 =================
def calculate_angle(a, b, c):
    """计算三个点构成的角度 (b 为顶点)"""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# ================= 视频读取线程 =================
class VideoThread(QThread):
    frame_ready = PySideSignal(QImage, int)
    finished = PySideSignal()

    def __init__(self, video_path, fps):
        super().__init__()
        self.video_path = video_path
        self.fps = fps if fps > 0 else 30
        self.running = True
        self.paused = False
        self.target_frame = -1

    def run(self):
        cap = None
        if self.video_path and os.path.exists(self.video_path):
            cap = cv2.VideoCapture(self.video_path)
            
        frame_idx = 0
        while self.running:
            if not self.paused:
                if cap and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_frame.shape
                    bytes_per_line = ch * w
                    q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    self.frame_ready.emit(q_img.copy(), frame_idx)
                else:
                    # 无视频模式，仅发送空图像和帧索引
                    self.frame_ready.emit(QImage(), frame_idx)
                    
                frame_idx += 1
                self.msleep(int(1000 / self.fps))
            else:
                self.msleep(50)
                
        if cap:
            cap.release()
        self.finished.emit()

    def pause(self): self.paused = True
    def resume(self): self.paused = False
    def stop(self): self.running = False

# ================= 绘制画布 =================
class PoseCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.background_image = None
        self.frame_data = None
        self.orig_w = 1920
        self.orig_h = 1080
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        self.kpt_names = [
            "nose", "L_eye", "R_eye", "L_ear", "R_ear",
            "L_sho", "R_sho", "L_elb", "R_elb",
            "L_wri", "R_wri", "L_hip", "R_hip",
            "L_kne", "R_kne", "L_ank", "R_ank"
        ]
        
        self.skeleton = [
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
            [5, 11], [6, 12], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
            [0, 1], [1, 3], [0, 2], [2, 4]
        ]
        
        # 需要计算角度的关节: (顶点索引, 点1索引, 点2索引, 显示名称)
        self.angles_to_calc = [
            (7, 5, 9, "L_Elbow"),
            (8, 6, 10, "R_Elbow"),
            (13, 11, 15, "L_Knee"),
            (14, 12, 16, "R_Knee")
        ]

    def set_background(self, qimg):
        if not qimg.isNull():
            self.background_image = QPixmap.fromImage(qimg)
        else:
            self.background_image = None
        self.update()

    def set_frame_data(self, df, orig_w, orig_h):
        self.frame_data = df
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.calculate_scale()
        self.update()

    def resizeEvent(self, event):
        self.calculate_scale()
        super().resizeEvent(event)

    def calculate_scale(self):
        w = self.width()
        h = self.height()
        if self.orig_w > 0 and self.orig_h > 0:
            self.scale = min(w / self.orig_w, h / self.orig_h)
            self.offset_x = (w - self.orig_w * self.scale) / 2
            self.offset_y = (h - self.orig_h * self.scale) / 2

    def map_point(self, x, y):
        return QPointF(x * self.scale + self.offset_x, y * self.scale + self.offset_y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 1. 绘制背景
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        if self.background_image and not self.background_image.isNull():
            # 修复 PySide6 drawPixmap 兼容性问题：使用 QRect 替代 QRectF，并转换为整数
            target_rect = QRect(int(self.offset_x), int(self.offset_y), 
                                int(self.orig_w * self.scale), int(self.orig_h * self.scale))
            painter.drawPixmap(target_rect, self.background_image)
            
        if self.frame_data is None or self.frame_data.empty:
            painter.end()
            return

        # 2. 绘制骨架和关键点
        pen = QPen()
        pen.setWidth(max(2, int(3 * self.scale)))
        font = QFont("Segoe UI", max(8, int(10 * self.scale)))
        painter.setFont(font)
        
        for _, row in self.frame_data.iterrows():
            xs = np.nan_to_num(row[[f"{n}_x" for n in self.kpt_names]].values, nan=0.0)
            ys = np.nan_to_num(row[[f"{n}_y" for n in self.kpt_names]].values, nan=0.0)
            confs = np.nan_to_num(row[[f"{n}_conf" for n in self.kpt_names]].values, nan=0.0)
            
            # 绘制连线
            for link in self.skeleton:
                i, j = link
                if confs[i] > 0.3 and confs[j] > 0.3:
                    pt1 = self.map_point(xs[i], ys[i])
                    pt2 = self.map_point(xs[j], ys[j])
                    pen.setColor(QColor(255, 255, 255, 180))
                    painter.setPen(pen)
                    painter.drawLine(pt1, pt2)
                    
            # 绘制圆点和名称
            for i in range(17):
                if confs[i] > 0.3:
                    pt = self.map_point(xs[i], ys[i])
                    radius = max(3, int(5 * self.scale))
                    
                    # 圆点
                    painter.setBrush(QColor(0, 220, 255))
                    painter.setPen(QPen(QColor(255, 255, 255), 1))
                    painter.drawEllipse(pt, radius, radius)
                    
                    # 名称
                    painter.setPen(QColor(255, 220, 0))
                    painter.drawText(pt + QPointF(radius + 4, -radius), self.kpt_names[i])

            # 3. 计算并绘制角度
            for vertex_idx, p1_idx, p2_idx, angle_name in self.angles_to_calc:
                if confs[vertex_idx] > 0.3 and confs[p1_idx] > 0.3 and confs[p2_idx] > 0.3:
                    v_pt = np.array([xs[vertex_idx], ys[vertex_idx]])
                    p1_pt = np.array([xs[p1_idx], ys[p1_idx]])
                    p2_pt = np.array([xs[p2_idx], ys[p2_idx]])
                    
                    angle = calculate_angle(p1_pt, v_pt, p2_pt)
                    text = f"{angle_name}: {angle:.0f}°"
                    v_qpt = self.map_point(v_pt[0], v_pt[1])
                    
                    # 绘制带背景的文本框
                    fm = QFontMetrics(font)
                    text_rect = fm.boundingRect(text)
                    bg_rect = QRectF(v_qpt.x() + 15, v_qpt.y() - 20, text_rect.width() + 8, text_rect.height() + 4)
                    painter.fillRect(bg_rect, QColor(0, 0, 0, 160))
                    
                    painter.setPen(QColor(255, 80, 80))
                    painter.drawText(bg_rect, Qt.AlignmentFlag.AlignCenter, text)

        painter.end()

# ================= 主窗口 =================
class PoseViewerWindow(QMainWindow):
    def __init__(self, df, meta, has_video):
        super().__init__()
        self.setWindowTitle("YOLO-Pose CSV Viewer")
        self.resize(1280, 720)
        
        self.df = df
        self.meta = meta
        self.has_video = has_video
        self.current_frame_idx = 0
        self.total_frames = int(meta.get("total_frames", df['frame_id'].max() + 1))
        self.fps = float(meta.get("fps", 30.0))
        
        self.orig_w = int(meta.get("width", 1920))
        self.orig_h = int(meta.get("height", 1080))

        self.init_ui()
        self.init_thread()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = PoseCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.canvas)

        # 控制栏
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(10, 10, 10, 10)
        
        self.btn_play = QPushButton("Pause")
        self.btn_play.setFixedWidth(80)
        self.btn_play.clicked.connect(self.toggle_play)
        control_layout.addWidget(self.btn_play)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, self.total_frames - 1))
        self.slider.valueChanged.connect(self.slider_changed)
        control_layout.addWidget(self.slider)

        self.lbl_frame = QLabel(f"Frame: 0 / {self.total_frames}")
        self.lbl_frame.setFixedWidth(150)
        self.lbl_frame.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        control_layout.addWidget(self.lbl_frame)

        layout.addLayout(control_layout)
        
        # 初始设置
        self.canvas.set_frame_data(self.df[self.df['frame_id'] == 0], self.orig_w, self.orig_h)

    def init_thread(self):
        video_path = self.meta.get("source_video") if self.has_video else None
        self.thread = VideoThread(video_path, self.fps)
        self.thread.frame_ready.connect(self.update_frame)
        self.thread.finished.connect(self.thread_finished)
        self.thread.start()

    def update_frame(self, q_img, frame_idx):
        self.current_frame_idx = frame_idx
        self.canvas.set_background(q_img)
        
        frame_data = self.df[self.df['frame_id'] == frame_idx]
        self.canvas.set_frame_data(frame_data, self.orig_w, self.orig_h)
        
        self.slider.blockSignals(True)
        self.slider.setValue(frame_idx)
        self.slider.blockSignals(False)
        self.lbl_frame.setText(f"Frame: {frame_idx} / {self.total_frames}")

    def slider_changed(self, value):
        pass

    def toggle_play(self):
        if self.thread.paused:
            self.thread.resume()
            self.btn_play.setText("Pause")
        else:
            self.thread.pause()
            self.btn_play.setText("Play")

    def thread_finished(self):
        self.btn_play.setText("Play")
        self.thread.pause()

    def closeEvent(self, event):
        self.thread.stop()
        self.thread.wait()
        event.accept()

# ================= CLI 命令 =================
@click.group()
def cli():
    """YOLO-Pose 数据处理与可视化工具集。"""
    pass

@cli.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('output_dir', type=click.Path(file_okay=False, dir_okay=True))
@click.option('--model', default='yolo11n-pose.pt', show_default=True, help='YOLO-Pose 模型权重路径或名称。')
@click.option('--imgsz', default=640, show_default=True, type=int, help='推理时的图像尺寸。')
@click.option('--subset', type=click.Choice(['normal', 'abnormal', 'test']), default='normal', show_default=True, help='数据集子集类型。')
def video2csv(input_dir, output_dir, model, imgsz, subset):
    """提取视频目录中的姿态数据并存储为 CSV 与 JSON 文件。
    
    SUBSET 参数指定输出到哪个子目录: normal(正常), abnormal(不规范), test(测试集)
    """
    # 根据 subset 创建对应的子目录
    subset_output_dir = os.path.join(output_dir, subset)
    os.makedirs(subset_output_dir, exist_ok=True)
    
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
        
    if not video_files:
        click.echo(f"未在目录 {input_dir} 中找到视频文件。")
        return

    click.echo(f"加载模型: {model}")
    yolo_model = YOLO(model)
    
    kpt_names = [
        "nose", "L_eye", "R_eye", "L_ear", "R_ear",
        "L_sho", "R_sho", "L_elb", "R_elb",
        "L_wri", "R_wri", "L_hip", "R_hip",
        "L_kne", "R_kne", "L_ank", "R_ank"
    ]
    
    columns = ["frame_id", "person_id"]
    for name in kpt_names:
        columns.extend([f"{name}_x", f"{name}_y", f"{name}_conf"])

    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_csv_path = os.path.join(subset_output_dir, f"{video_name}.csv")
        output_json_path = os.path.join(subset_output_dir, f"{video_name}.json")

        # 检查 CSV 和 JSON 是否都已存在
        if os.path.exists(output_csv_path) and os.path.exists(output_json_path):
            click.echo(f"跳过 (已存在): {video_name}")
            continue

        click.echo(f"正在处理视频: {video_name} ...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            click.echo(f"错误：无法打开视频 {video_path}")
            continue
            
        metadata = {
            "source_video": os.path.abspath(video_path),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        
        frame_id = 0
        all_data = []
        
        while True:
            ret, frame = cap.read()
            if not ret: break
                
            results = yolo_model(frame, verbose=False, imgsz=imgsz)
            result = results[0]
            
            if result.keypoints is not None and result.keypoints.data.shape[0] > 0:
                keypoints_xy = result.keypoints.xy.cpu().numpy()
                keypoints_conf = result.keypoints.conf.cpu().numpy()
                num_persons = keypoints_xy.shape[0]
                
                for person_id in range(num_persons):
                    row = [frame_id, person_id]
                    for i in range(17):
                        row.extend([
                            keypoints_xy[person_id, i, 0],
                            keypoints_xy[person_id, i, 1],
                            keypoints_conf[person_id, i]
                        ])
                    all_data.append(row)
            frame_id += 1
            
        cap.release()
        
        df = pd.DataFrame(all_data, columns=columns) if all_data else pd.DataFrame(columns=columns)
        df.to_csv(output_csv_path, index=False)
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
            
        click.echo(f"完成: {output_csv_path} 与 {output_json_path}")

@cli.command()
@click.argument('csv_file', type=click.Path(exists=True, dir_okay=False))
def csvshow(csv_file):
    """加载指定的 CSV 文件并以 PySide6 界面展示姿态骨架。"""
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        click.echo(f"错误：无法读取 CSV 文件。{e}")
        return

    if df.empty:
        click.echo("CSV 文件为空，无数据可展示。")
        return

    csv_dir = os.path.dirname(csv_file)
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    json_path = os.path.join(csv_dir, f"{base_name}.json")
    
    meta = {}
    has_video = False
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        video_path = meta.get("source_video")
        if video_path and os.path.exists(video_path):
            has_video = True
    else:
        meta = {
            "width": 1920, "height": 1080, "fps": 30.0,
            "total_frames": df['frame_id'].max() + 1
        }

    app = QApplication(sys.argv)
    window = PoseViewerWindow(df, meta, has_video)
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    cli()