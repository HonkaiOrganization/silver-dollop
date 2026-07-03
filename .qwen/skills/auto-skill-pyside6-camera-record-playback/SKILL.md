---
name: pyside6-camera-record-playback
description: PySide6摄像机录制+回放架构：CameraThread增量写MP4/CSV，PlaybackThread同步回放相机帧与骨架渲染
source: auto-skill
extracted_at: '2026-07-03T05:20:39.202Z'
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

### CSV 格式（与 PoseExtractor 对齐）

```python
KPT_NAMES = ["nose", "L_eye", "R_eye", "L_ear", "R_ear",
             "L_sho", "R_sho", "L_elb", "R_elb",
             "L_wri", "R_wri", "L_hip", "R_hip",
             "L_kne", "R_kne", "L_ank", "R_ank"]

CSV_COLUMNS = ["frame_id", "person_id"] + [
    f"{name}_{suffix}" for name in KPT_NAMES for suffix in ("x", "y", "conf")
]
# 总计 53 列: frame_id, person_id, 17×(x, y, conf)
```

### 信号设计

```python
frames_ready = Signal(object, object)           # (camera_bgr, skeleton_bgr) 实时预览
recording_progress = Signal(float, float)        # (elapsed_sec, frames_written)
recording_saved = Signal(str, str, int, float)   # (video_path, csv_path, total_frames, fps)
recording_too_short = Signal()                   # 录制不足 10 秒
```

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

## MainWindow UI 模式切换

```
录制模式:  [摄像机选择▼]  [● 开始记录]  [■ 结束记录]  状态文字
回放模式:  [▶ 播放]  [═══════ 进度条 ═══════]  0/300  [提交分析]
```

切换逻辑：
1. 录制完成 → `recording_bar.hide()` + `playback_bar.show()`
2. 停止 CameraThread，关闭摄像机
3. 启动 PlaybackThread，连接 `frames_ready` 信号
4. 提交分析 → 切换到 AnalysisPage（详见 `pyside6-analysis-page` skill）
