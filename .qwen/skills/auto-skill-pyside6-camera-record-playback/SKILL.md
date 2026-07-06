---
name: pyside6-camera-record-playback
description: PySide6 camera recording + playback + file import architecture — CameraThread/FileImportThread with incremental MP4/CSV writing, PlaybackThread synchronized playback, real-time keypoint coordinate table display, menu bar (File/Help), native PySide styling
source: auto-skill
extracted_at: '2026-07-04T12:00:00.000Z'
---

# PySide6 Camera Recording and Playback Architecture

Applicable to desktop GUI applications that follow a "record first, preview via playback, then submit for analysis" workflow.

## Overall Architecture

```
Recording Phase                         Playback Phase
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

Both phases are managed by the same MainWindow, switching UI modes by hiding/showing the control bar.

## CameraThread (Recording)

### Incremental MP4 Writing (Avoiding Memory Bloat)

Do not buffer all frames in memory during recording; use `cv2.VideoWriter` to write frame by frame:

```python
def start_recording(self, video_path, csv_path, fps=30.0):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = self.skeleton_target_size  # (1080, 1920)
    self._video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    self._csv_rows = []
    self._record_frame_id = 0

def _record_frame(self, processed_camera, pose_result):
    # Resize camera frame to match skeleton dimensions before writing to MP4
    cam_resized = cv2.resize(processed_camera, (w, h))
    self._video_writer.write(cam_resized)

    # Buffer keypoint rows into list
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

### CSV Format (Unified Definition in config.py)

All modules share a single set of constant definitions (`config.py`) to avoid duplication:

```python
# config.py
KPT_NAMES = ["nose", "L_eye", "R_eye", "L_ear", "R_ear",
             "L_sho", "R_sho", "L_elb", "R_elb",
             "L_wri", "R_wri", "L_hip", "R_hip",
             "L_kne", "R_kne", "L_ank", "R_ank"]

CSV_COLUMNS = ["frame_id", "person_id"] + [
    f"{name}_{suffix}" for name in KPT_NAMES for suffix in ("x", "y", "conf")
]
# Total 53 columns: frame_id, person_id, 17x(x, y, conf)

SKELETON_LINKS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (0, 1), (1, 3), (0, 2), (2, 4),
]
```

Usage: `from config import KPT_NAMES, CSV_COLUMNS, SKELETON_LINKS`

### Signal Design

```python
frames_ready = Signal(object, object)           # (camera_bgr, skeleton_bgr) live preview
keypoints_ready = Signal(object, object)        # (xy: [17,2], conf: [17]) or (None, None) — for real-time coordinate table + inference updates
recording_progress = Signal(float, float)        # (elapsed_sec, frames_written)
recording_saved = Signal(str, str, int, float)   # (video_path, csv_path, total_frames, fps)
recording_too_short = Signal()                   # Recording duration less than 10 seconds
```

`keypoints_ready` emits `(xy, conf)` after each frame's inference, carrying confidence scores for downstream filtering. Both recording and playback share the same slot to update the QTableWidget.

## PlaybackThread (Playback)

### CSV Pre-Indexing (Critical Performance Optimization)

Do not filter with `df[df['frame_id'] == idx]` for every frame (O(N) per frame); build an index list once at the beginning of `run()`:

```python
def _build_frame_index(self, df, total_frames):
    """Each element is (xy[17,2], conf[17]) or (None, None)"""
    index = [(None, None)] * total_frames
    grouped = df.groupby("frame_id")
    for fid, group in grouped:
        fid = int(fid)
        if fid >= total_frames: continue
        # When multiple persons exist, select the person_id with the highest average confidence
        row = group.iloc[0]  # or sort by person_id
        xy = np.stack([row[x_cols].values, row[y_cols].values], axis=1)
        conf = row[c_cols].values.astype(np.float64)
        index[fid] = (xy, conf)
    return index
```

### Playback Loop

```python
# Signals
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

        # Render skeleton from pre-built index
        skeleton = np.zeros((h, w, 3), dtype=np.uint8)
        if frame_idx < len(frame_kpts) and frame_kpts[frame_idx][0] is not None:
            xy, conf = frame_kpts[frame_idx]
            PoseProcessor.render_skeleton(skeleton, xy, conf)

        self.frames_ready.emit(camera_frame, skeleton, frame_idx)
        frame_idx += 1
        self.msleep(frame_interval_ms)
```

### Seek + Play/Pause Control

```python
# External calls
def play(self):  self._paused = False
def pause(self): self._paused = True
def seek(self, frame_idx): self._seek_to = frame_idx  # processed on next loop iteration
```

Set `_slider_dragging = True` during slider drag to suppress frame number writeback; call `seek()` on release.

## FileImportThread (External Video Import)

Bypasses camera recording; performs pose inference frame by frame on an external video file, producing MP4 + CSV output identical to recording, then reuses PlaybackThread for playback.

### Architecture

```
External video file (*.mp4/avi/mkv/...)
        │
        ▼
FileImportThread (QThread)
  ├── cv2.VideoCapture frame-by-frame reading
  ├── FrameProcessor.process() 9:16 cropping
  ├── PoseProcessor.process() pose inference
  ├── cv2.VideoWriter writes MP4 (same 1080x1920 dimensions as CameraThread)
  └── CSV row collection (same CSV_COLUMNS as CameraThread)
        │
        ▼ finished(video_path, csv_path, total_frames, fps)
        │
        ▼
_switch_to_playback() → PlaybackThread → playback/analysis
```

### Key Implementation

```python
class FileImportThread(QThread):
    progress = Signal(int, int)            # (current_frame, total_frames)
    finished = Signal(str, str, int, float) # (video_path, csv_path, total_frames, fps)
    error = Signal(str)

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Output MP4 (identical dimensions and encoding as CameraThread)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video, fourcc, fps, (1080, 1920))

        csv_rows = []
        frame_id = 0
        while not self._stop_requested:
            ret, frame = cap.read()
            if not ret: break

            processed = FrameProcessor.process(frame)
            pose_result = self.pose_processor.process(processed, target_size=(1080, 1920))

            # Write to MP4 (same logic as CameraThread._record_frame)
            cam_resized = cv2.resize(processed, (1080, 1920))
            writer.write(cam_resized)

            # Collect CSV rows (same CSV_COLUMNS as CameraThread)
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

        # Write CSV (same format as CameraThread._finish_recording)
        df = pd.DataFrame(csv_rows, columns=CSV_COLUMNS)
        df.to_csv(output_csv, index=False)

        self.finished.emit(output_video, output_csv, frame_id, fps)
```

### Consistency with CameraThread

| Item | CameraThread | FileImportThread |
|------|-------------|-----------------|
| Frame source | `camera_manager.read_frame()` | `cv2.VideoCapture.read()` |
| Frame processing | `FrameProcessor.process()` | Same |
| Pose inference | `pose_processor.process()` | Same |
| MP4 encoding | `mp4v`, 1080x1920 | Same |
| CSV format | `CSV_COLUMNS` (53 columns) | Same |
| Output signal | `recording_saved(path, path, frames, fps)` | `finished(path, path, frames, fps)` |

Both produce identical output; PlaybackThread does not need to distinguish the source.

### MainWindow Integration

```python
def _on_open_video_file(self):
    file_path, _ = QFileDialog.getOpenFileName(
        self, "Open Video File", "",
        "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv *.flv);;All Files (*)"
    )
    if not file_path: return

    # Stop current camera
    if self.camera_thread:
        self.camera_thread.stop()
        self.camera_thread = None
    self.camera_manager.close_camera()

    # Generate output paths
    uid = uuid.uuid4().hex[:8]
    video_out = os.path.join(TEMP_DIR, f"imp_{uid}.mp4")
    csv_out = os.path.join(TEMP_DIR, f"imp_{uid}.csv")

    # Progress dialog
    progress = QProgressDialog("Importing...", "Cancel", 0, 100, self)
    progress.setWindowModality(Qt.WindowModality.WindowModal)

    # Start import thread
    self._import_thread = FileImportThread(file_path, video_out, csv_out, self.pose_processor)
    self._import_thread.progress.connect(lambda c, t: (progress.setMaximum(t), progress.setValue(c)))
    self._import_thread.finished.connect(self._on_import_finished)
    self._import_thread.error.connect(self._on_import_error)
    progress.canceled.connect(self._import_thread.stop)
    self._import_thread.start()

def _on_import_finished(self, video_path, csv_path, total_frames, fps):
    # Directly reuse the playback switch logic from recording completion
    self._switch_to_playback(video_path, csv_path, total_frames, fps)
```

**Key point**: `_on_import_finished` directly calls `_switch_to_playback()`, reusing existing playback logic with no additional code required.

### closeEvent Cleanup

```python
def closeEvent(self, event):
    if self._import_thread and self._import_thread.isRunning():
        self._import_thread.stop()
    # ... other cleanup ...
```

## MainWindow Menu Bar

Add the menu bar at the beginning of `init_ui()`:

```python
menu_bar = self.menuBar()

# File menu
file_menu = menu_bar.addMenu("File(&F)")
act_open = file_menu.addAction("Open Video File(&O)...")
act_open.setShortcut("Ctrl+O")
act_open.triggered.connect(self._on_open_video_file)
file_menu.addSeparator()
act_exit = file_menu.addAction("Exit(&X)")
act_exit.setShortcut("Ctrl+Q")
act_exit.triggered.connect(self.close)

# Help menu
help_menu = menu_bar.addMenu("Help(&H)")
act_help = help_menu.addAction("View Help Documentation(&H)")
act_help.setShortcut("F1")
act_help.triggered.connect(self._on_show_help)
```

Open README.md with `webbrowser.open()` for the help documentation:

```python
def _on_show_help(self):
    readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md")
    if os.path.exists(readme_path):
        webbrowser.open(readme_path)
```

## MainWindow UI Mode Switching

```
Recording mode:  [Camera select v]  [● Start Recording]  [Stop Recording]  Status text
Playback mode:   [> Play]  [========= Progress bar =========]  0/300  [Submit Analysis]
```

Switching logic:
1. Recording complete **or** file import complete -> `recording_bar.hide()` + `playback_bar.show()`
2. Stop CameraThread, close camera
3. Start PlaybackThread, connect `frames_ready` signal
4. Submit analysis -> switch to AnalysisPage (see `pyside6-analysis-page` skill for details)

## Keypoint Coordinate Table (Real-Time X/Y Display)

Add a QTableWidget to the right of the dual view (camera + skeleton). During recording, it displays 17 keypoint coordinates in real time; during playback, it shows the current frame's coordinates.

### Layout

```
+------------------+------------------+------------+
|  Camera Feed     |  Pose Skeleton   | Keypoint   |
|  (QLabel)        |  (QLabel)        | Table      |
|                  |                  | Name|X|Y   |
|                  |                  | 17 rows    |
+------------------+------------------+------------+
```

### Table Initialization

```python
from config import KPT_NAMES

self.kpt_table = QTableWidget(17, 3)
self.kpt_table.setHorizontalHeaderLabels(["Keypoint", "X", "Y"])
self.kpt_table.verticalHeader().setVisible(False)
self.kpt_table.setFixedWidth(280)
self.kpt_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
self.kpt_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
# Three columns evenly distributed
for col in range(3):
    self.kpt_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)

for i, name in enumerate(KPT_NAMES):
    item = QTableWidgetItem(name)
    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    self.kpt_table.setItem(i, 0, item)
    self.kpt_table.setItem(i, 1, QTableWidgetItem("-"))
    self.kpt_table.setItem(i, 2, QTableWidgetItem("-"))
```

### Unified Update Slot (Shared by Recording + Playback)

Both CameraThread and PlaybackThread emit the `keypoints_ready` signal, connected to the same slot:

```python
# Connections (once each in init_camera_system and _switch_to_playback)
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

Clear the table when switching back to recording mode:
```python
for i in range(17):
    self.kpt_table.item(i, 1).setText("-")
    self.kpt_table.item(i, 2).setText("-")
```
