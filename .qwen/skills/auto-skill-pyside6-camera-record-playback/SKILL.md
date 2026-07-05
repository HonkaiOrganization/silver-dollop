---
name: pyside6-camera-record-playback
description: PySide6摄像机录制+回放+文件导入架构：CameraThread/FileImportThread增量写MP4/CSV，PlaybackThread同步回放，关键点坐标表格实时显示，菜单栏（文件/帮助），原生PySide风格
source: auto-skill
extracted_at: '2026-07-04T12:00:00.000Z'
---

# PySide6 摄像机录制与回放架构

适用于需要"先录制、再回放预览、最后提交分析"的桌面 GUI 应用。

## 整体架构

```
录制阶段                              回放阶段
┌──────────────┐   save    ┌────────────────────────┐
│ CameraThread  │ ───────► │ temp/rec_<uid>.mp4      │
│  (live cam +  │          │ temp/rec_<uid>.csv      │
│   inference)  │          └───────────┬────────────┘
└──────────────┘                      │
                                      ▼
                            ┌──────────────────┐
                            │ PlaybackThread    │
                            │  (MP4 + CSV sync) │
                            └──────────────────┘
```

两阶段由同一个 MainWindow 管理，通过隐藏/显示控制栏切换 UI 模式。

## CameraThread（录制）

### MP4 增量写入（避免内存爆炸）

录制期间不要缓存所有帧到内存，用 `cv2.VideoWriter` 逐帧写入：

```python
def start_recording(self, video_path, csv_path, fps=30.0):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = self.skeleton_target_size  # (1080, 1920)
    self._video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    self._csv_rows = []
    self._record_frame_id = 0

def _record_frame(self, processed_camera, pose_result):
    # 相机帧缩放至与骨架同尺寸后写入 MP4
    cam_resized = cv2.resize(processed_camera, (w, h))
    self._video_writer.write(cam_resized)

    # 关键点行缓存到列表
    xy, conf = pose_result["keypoints_array"]["xy"], ...
    row = [float(self._record_frame_id), 0.0]
    for i in range(17):
        row.extend([float(xy[i,0]), float(xy[i,1]), float(conf[i])])
    self._csv_rows.append(row)

def _finish_recording(self):
    self._video_writer.release()
    df = pd.DataFrame(self._csv_rows, columns=CSV_COLUMNS)
    df.to_csv(csv_path, index=False)
```

### CSV 格式（统一定义在 config.py）

所有模块共享同一份常量定义（`config.py`），避免重复：

```python
# config.py
KPT_NAMES = ["nose", "L_eye", "R_eye", "L_ear", "R_ear",
             "L_sho", "R_sho", "L_elb", "R_elb",
             "L_wri", "R_wri", "L_hip", "R_hip",
             "L_kne", "R_kne", "L_ank", "R_ank"]

CSV_COLUMNS = ["frame_id", "person_id"] + [
    f"{name}_{suffix}" for name in KPT_NAMES for suffix in ("x", "y", "conf")
]
# 总计 53 列: frame_id, person_id, 17×(x, y, conf)

SKELETON_LINKS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (0, 1), (1, 3), (0, 2), (2, 4),
]
```

使用方式：`from config import KPT_NAMES, CSV_COLUMNS, SKELETON_LINKS`

### 信号设计

```python
frames_ready = Signal(object, object)           # (camera_bgr, skeleton_bgr) 实时预览
keypoints_ready = Signal(object, object)        # (xy: [17,2], conf: [17]) or (None, None) — 用于实时更新坐标表格 + 推理
recording_progress = Signal(float, float)        # (elapsed_sec, frames_written)
recording_saved = Signal(str, str, int, float)   # (video_path, csv_path, total_frames, fps)
recording_too_short = Signal()                   # 录制不足 10 秒
```

`keypoints_ready` 在每帧推理后发射 `(xy, conf)`，携带置信度用于下游过滤。录制和回放共用同一 slot 更新 QTableWidget。

## PlaybackThread（回放）

### CSV 预索引（关键性能优化）

不要每帧都用 `df[df['frame_id'] == idx]` 过滤（O(N) 每帧），在 `run()` 开头一次构建索引列表：

```python
def _build_frame_index(self, df, total_frames):
    """每个元素为 (xy[17,2], conf[17]) 或 (None, None)"""
    index = [(None, None)] * total_frames
    grouped = df.groupby("frame_id")
    for fid, group in grouped:
        fid = int(fid)
        if fid >= total_frames: continue
        # 多人时取平均置信度最高的 person_id
        row = group.iloc[0]  # 或按 person_id 排序
        xy = np.stack([row[x_cols].values, row[y_cols].values], axis=1)
        conf = row[c_cols].values.astype(np.float64)
        index[fid] = (xy, conf)
    return index
```

### 回放循环

```python
# 信号
frames_ready = Signal(object, object, int)  # (camera_bgr, skeleton_bgr, frame_idx)
keypoints_ready = Signal(object, object)    # (xy: [17,2], conf: [17]) or (None, None)
playback_finished = Signal()

def run(self):
    cap = cv2.VideoCapture(self.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_kpts = self._build_frame_index(df, total_frames)
    frame_interval_ms = int(1000 / fps)

    while self._is_running:
        if self._seek_to >= 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self._seek_to)
            frame_idx = self._seek_to
            self._seek_to = -1

        if self._paused:
            self.msleep(50)
            continue

        ret, camera_frame = cap.read()
        if not ret:
            self.playback_finished.emit()
            break

        # 从预建索引渲染骨架
        skeleton = np.zeros((h, w, 3), dtype=np.uint8)
        if frame_idx < len(frame_kpts) and frame_kpts[frame_idx][0] is not None:
            xy, conf = frame_kpts[frame_idx]
            PoseProcessor.render_skeleton(skeleton, xy, conf)

        self.frames_ready.emit(camera_frame, skeleton, frame_idx)
        frame_idx += 1
        self.msleep(frame_interval_ms)
```

### seek + play/pause 控制

```python
# 外部调用
def play(self):  self._paused = False
def pause(self): self._paused = True
def seek(self, frame_idx): self._seek_to = frame_idx  # 下一循环处理
```

Slider 拖拽时设置 `_slider_dragging = True` 暂停帧号回写，松手时调 `seek()`。

## FileImportThread（外部视频导入）

跳过摄像机录制，从外部视频文件逐帧执行姿态推理，生成与录制完全一致的 MP4 + CSV 输出，然后复用 PlaybackThread 回放。

### 架构

```
外部视频文件 (*.mp4/avi/mkv/...)
        │
        ▼
FileImportThread (QThread)
  ├── cv2.VideoCapture 逐帧读取
  ├── FrameProcessor.process() 裁剪 9:16
  ├── PoseProcessor.process() 姿态推理
  ├── cv2.VideoWriter 写 MP4（与 CameraThread 同尺寸 1080×1920）
  └── CSV 行收集（与 CameraThread 同 CSV_COLUMNS）
        │
        ▼ finished(video_path, csv_path, total_frames, fps)
        │
        ▼
_switch_to_playback() → PlaybackThread → 回放/分析
```

### 关键实现

```python
class FileImportThread(QThread):
    progress = Signal(int, int)            # (current_frame, total_frames)
    finished = Signal(str, str, int, float) # (video_path, csv_path, total_frames, fps)
    error = Signal(str)

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # 输出 MP4（与 CameraThread 完全一致的尺寸和编码）
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video, fourcc, fps, (1080, 1920))

        csv_rows = []
        frame_id = 0
        while not self._stop_requested:
            ret, frame = cap.read()
            if not ret: break

            processed = FrameProcessor.process(frame)
            pose_result = self.pose_processor.process(processed, target_size=(1080, 1920))

            # 写入 MP4（与 CameraThread._record_frame 同逻辑）
            cam_resized = cv2.resize(processed, (1080, 1920))
            writer.write(cam_resized)

            # 收集 CSV 行（与 CameraThread 同 CSV_COLUMNS）
            xy, conf = pose_result["keypoints_array"]["xy"], ...
            if xy is not None:
                row = [float(frame_id), 0.0]
                for i in range(17):
                    row.extend([float(xy[i,0]), float(xy[i,1]), float(conf[i])])
                csv_rows.append(row)

            frame_id += 1
            if frame_id % 10 == 0:
                self.progress.emit(frame_id, total_frames)

        cap.release()
        writer.release()

        # 写 CSV（与 CameraThread._finish_recording 同格式）
        df = pd.DataFrame(csv_rows, columns=CSV_COLUMNS)
        df.to_csv(output_csv, index=False)

        self.finished.emit(output_video, output_csv, frame_id, fps)
```

### 与 CameraThread 的一致性

| 项目 | CameraThread | FileImportThread |
|------|-------------|-----------------|
| 帧来源 | `camera_manager.read_frame()` | `cv2.VideoCapture.read()` |
| 帧处理 | `FrameProcessor.process()` | 同 |
| 姿态推理 | `pose_processor.process()` | 同 |
| MP4 编码 | `mp4v`, 1080×1920 | 同 |
| CSV 格式 | `CSV_COLUMNS` (53列) | 同 |
| 输出信号 | `recording_saved(path, path, frames, fps)` | `finished(path, path, frames, fps)` |

两者输出完全一致，PlaybackThread 无需区分来源。

### MainWindow 集成

```python
def _on_open_video_file(self):
    file_path, _ = QFileDialog.getOpenFileName(
        self, "打开视频文件", "",
        "视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv *.flv);;所有文件 (*)"
    )
    if not file_path: return

    # 停止当前摄像机
    if self.camera_thread:
        self.camera_thread.stop()
        self.camera_thread = None
    self.camera_manager.close_camera()

    # 生成输出路径
    uid = uuid.uuid4().hex[:8]
    video_out = os.path.join(TEMP_DIR, f"imp_{uid}.mp4")
    csv_out = os.path.join(TEMP_DIR, f"imp_{uid}.csv")

    # 进度对话框
    progress = QProgressDialog("正在导入…", "取消", 0, 100, self)
    progress.setWindowModality(Qt.WindowModality.WindowModal)

    # 启动导入线程
    self._import_thread = FileImportThread(file_path, video_out, csv_out, self.pose_processor)
    self._import_thread.progress.connect(lambda c, t: (progress.setMaximum(t), progress.setValue(c)))
    self._import_thread.finished.connect(self._on_import_finished)
    self._import_thread.error.connect(self._on_import_error)
    progress.canceled.connect(self._import_thread.stop)
    self._import_thread.start()

def _on_import_finished(self, video_path, csv_path, total_frames, fps):
    # 直接复用录制完成后的回放切换逻辑
    self._switch_to_playback(video_path, csv_path, total_frames, fps)
```

**关键点**：`_on_import_finished` 直接调用 `_switch_to_playback()`，复用已有回放逻辑，无需额外代码。

### closeEvent 清理

```python
def closeEvent(self, event):
    if self._import_thread and self._import_thread.isRunning():
        self._import_thread.stop()
    # ... 其他清理 ...
```

## MainWindow 菜单栏

在 `init_ui()` 开头添加菜单栏：

```python
menu_bar = self.menuBar()

# 文件菜单
file_menu = menu_bar.addMenu("文件(&F)")
act_open = file_menu.addAction("打开视频文件(&O)…")
act_open.setShortcut("Ctrl+O")
act_open.triggered.connect(self._on_open_video_file)
file_menu.addSeparator()
act_exit = file_menu.addAction("退出(&X)")
act_exit.setShortcut("Ctrl+Q")
act_exit.triggered.connect(self.close)

# 帮助菜单
help_menu = menu_bar.addMenu("帮助(&H)")
act_help = help_menu.addAction("查看帮助文档(&H)")
act_help.setShortcut("F1")
act_help.triggered.connect(self._on_show_help)
```

帮助文档用 `webbrowser.open()` 打开 README.md：

```python
def _on_show_help(self):
    readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md")
    if os.path.exists(readme_path):
        webbrowser.open(readme_path)
```

## MainWindow UI 模式切换

```
录制模式:  [摄像机选择▼]  [● 开始记录]  [■ 结束记录]  状态文字
回放模式:  [▶ 播放]  [═══════ 进度条 ═══════]  0/300  [提交分析]
```

切换逻辑：
1. 录制完成 **或** 文件导入完成 → `recording_bar.hide()` + `playback_bar.show()`
2. 停止 CameraThread，关闭摄像机
3. 启动 PlaybackThread，连接 `frames_ready` 信号
4. 提交分析 → 切换到 AnalysisPage（详见 `pyside6-analysis-page` skill）

## 关键点坐标表格（实时 X/Y 显示）

在双视图（camera + skeleton）右侧添加 QTableWidget，录制时实时显示 17 关键点坐标，回放时显示当前帧坐标。

### 布局

```
┌──────────────────┬──────────────────┬────────────┐
│  Camera Feed     │  Pose Skeleton   │ 关键点表格  │
│  (QLabel)        │  (QLabel)        │ QTableWidget│
│                  │                  │ 名称|X|Y   │
│                  │                  │ 17行        │
└──────────────────┴──────────────────┴────────────┘
```

### 表格初始化

```python
from config import KPT_NAMES

self.kpt_table = QTableWidget(17, 3)
self.kpt_table.setHorizontalHeaderLabels(["关键点", "X", "Y"])
self.kpt_table.verticalHeader().setVisible(False)
self.kpt_table.setFixedWidth(280)
self.kpt_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
self.kpt_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
# 三列均分
for col in range(3):
    self.kpt_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)

for i, name in enumerate(KPT_NAMES):
    item = QTableWidgetItem(name)
    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    self.kpt_table.setItem(i, 0, item)
    self.kpt_table.setItem(i, 1, QTableWidgetItem("-"))
    self.kpt_table.setItem(i, 2, QTableWidgetItem("-"))
```

### 统一更新 slot（录制 + 回放共用）

CameraThread 和 PlaybackThread 都发射 `keypoints_ready` 信号，连接到同一个 slot：

```python
# 连接（init_camera_system 和 _switch_to_playback 各连一次）
self.camera_thread.keypoints_ready.connect(self._on_keypoints_updated)
self.playback_thread.keypoints_ready.connect(self._on_keypoints_updated)

def _on_keypoints_updated(self, xy):
    for i in range(17):
        if xy is not None:
            x_val = f"{xy[i, 0]:.1f}"
            y_val = f"{xy[i, 1]:.1f}"
        else:
            x_val = "-"
            y_val = "-"
        self.kpt_table.item(i, 1).setText(x_val)
        self.kpt_table.item(i, 2).setText(y_val)
```

切回录制模式时清空表格：
```python
for i in range(17):
    self.kpt_table.item(i, 1).setText("-")
    self.kpt_table.item(i, 2).setText("-")
```
