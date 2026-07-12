---
name: pyside6-analysis-page
description: PySide6 Analysis Page Architecture: QStackedWidget page switching + QThread async inference/VLM + matplotlib chart embedding + mosaic keyframe image cards (no video player), native PySide style
source: auto-skill
extracted_at: '2026-07-04T12:00:00.000Z'
---

# PySide6 Analysis Page Architecture

Applies to the analysis phase in the "Record -> Playback -> Submit Analysis" pipeline. After playback finishes, clicking "Submit Analysis" switches to the analysis page via QStackedWidget, automatically runs inference and displays charts, then manually triggers VLM deep analysis (card-style report with embedded video clips).

## File Structure

```
gui/
├── analysis_page.py          # AnalysisPage main class (page layout + logic)
├── widgets/
│   ├── __init__.py           # Exports SectionCard, FrameDisplay, VideoDisplay, SkeletonDisplay
│   ├── frame_display.py      # FrameDisplay base widget (BGR→QPixmap + resize)
│   ├── video_display.py      # VideoDisplay + SkeletonDisplay subclasses
│   └── section_card.py       # SectionCard VLM analysis card (mosaic images)
└── workers/
    ├── __init__.py           # Exports InferenceWorker, VLMWorker
    ├── inference_worker.py   # Inference background thread
    └── vlm_worker.py         # VLM analysis background thread
```

## Overall Architecture

```
MainWindow (QStackedWidget)
├── _main_page  (Record/Playback view)
└── _analysis_page  (AnalysisPage widget)
     ├── QThread: InferenceWorker -> Inference -> Charts + Statistics
     └── QThread: VLMWorker -> VLM Analysis -> SectionCard list (with mosaic keyframe images)
```

## MainWindow Integration

### QStackedWidget Page Switching

```python
from PySide6.QtWidgets import QStackedWidget

self._stack = QStackedWidget()
self.setCentralWidget(self._stack)

self._main_page = QWidget()
main_layout = QVBoxLayout(self._main_page)
# ... add original record/playback controls to main_layout ...
self._stack.addWidget(self._main_page)

self._analysis_page: AnalysisPage | None = None
```

### Submit Analysis -> Switch Page

```python
def _on_submit_analysis(self):
    if self.playback_thread:
        self.playback_thread.pause()

    if self._analysis_page is not None:
        self._analysis_page.cleanup()
        self._stack.removeWidget(self._analysis_page)
        self._analysis_page.deleteLater()

    self._analysis_page = AnalysisPage(
        self._playback_video_path,
        self._playback_csv_path,
        parent=self,
    )
    self._analysis_page.back_requested.connect(self._on_back_from_analysis)
    self._stack.addWidget(self._analysis_page)
    self._stack.setCurrentWidget(self._analysis_page)
    self._analysis_page.start_analysis()

def _on_back_from_analysis(self):
    if self._analysis_page is not None:
        self._analysis_page.cleanup()
        self._stack.removeWidget(self._analysis_page)
        self._analysis_page.deleteLater()
        self._analysis_page = None
    self._stack.setCurrentWidget(self._main_page)
```

### closeEvent Cleanup

```python
def closeEvent(self, event):
    # ... original camera_thread / playback_thread cleanup ...
    if self._analysis_page:
        self._analysis_page.cleanup()
    super().closeEvent(event)
```

## AnalysisPage Components

### QThread Worker Pattern

Both inference and VLM calls are time-consuming operations and must be offloaded to background threads. Use the `abandon` flag to prevent callbacks after the page is destroyed:

```python
class InferenceWorker(QThread):
    finished = Signal(dict)

    def __init__(self, inference, csv_path, output_json):
        super().__init__()
        self._inference = inference
        self._csv_path = csv_path
        self._output_json = output_json
        self._abandoned = False

    def abandon(self):
        """Called before page destruction to prevent the finished signal from firing"""
        self._abandoned = True

    def run(self):
        result = self._inference.predict(self._csv_path, self._output_json)
        if not self._abandoned:
            self.finished.emit(result)


class VLMWorker(QThread):
    finished = Signal(dict)  # Structured result, not str
    error = Signal(str)

    def run(self):
        try:
            result = analyze_windows(self.video_path, self.json_path, top_k=self.top_k)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
```

### Cleanup Method

```python
def cleanup(self):
    if self._inference_worker and self._inference_worker.isRunning():
        self._inference_worker.abandon()
        self._inference_worker.quit()
        self._inference_worker.wait(2000)
    if self._vlm_worker and self._vlm_worker.isRunning():
        self._vlm_worker.quit()
        self._vlm_worker.wait(5000)
    self._clear_sections()  # Clean up VideoPlayer temporary files
```

### matplotlib Chart Embedding

Use `FigureCanvasQTAgg` to embed matplotlib charts into the Qt layout. Use the system default color scheme (native style):

```python
matplotlib.use('Agg')  # Must be set before importing pyplot
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

def _plot_chart(self, r: dict):
    details = r.get("window_details", [])
    if not details:
        return

    # Clean up old charts (avoid memory leaks)
    for i in reversed(range(self._chart_layout.count())):
        w = self._chart_layout.itemAt(i).widget()
        if w:
            w.deleteLater()

    fig = Figure(figsize=(8, 3.5), dpi=100)  # Use default background color
    ax = fig.add_subplot(111)

    x = [d["start_frame"] for d in details]
    y = [d["prob_abnormal"] for d in details]

    ax.plot(x, y, marker='.', linewidth=1.2, markersize=4)
    ax.fill_between(x, y, alpha=0.15)
    ax.set_xlabel("Frame")
    ax.set_ylabel("P(abnormal)")
    ax.set_title("Per-Window Anomaly Confidence")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    canvas = FigureCanvas(fig)
    self._chart_layout.addWidget(canvas)
```

## Mosaic Keyframe Display (replaces VideoPlayer)

Instead of embedding a video player, extract keyframes from the base64 clip and display them as a grid mosaic image. This is simpler, avoids QMediaPlayer complexity, and matches the Word export visual style.

```python
import cv2, numpy as np, base64, os, uuid
from PySide6.QtGui import QImage, QPixmap

def _extract_keyframes(clip_b64: str, max_frames: int = 8) -> list[np.ndarray]:
    """base64 -> temp MP4 -> cv2.VideoCapture -> uniform frame sampling"""
    data = base64.b64decode(clip_b64)
    tmp_path = os.path.join(TEMP_DIR, f"clip_{uuid.uuid4().hex[:8]}.mp4")
    try:
        with open(tmp_path, 'wb') as f:
            f.write(data)
        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // max_frames)
        frames = []
        for idx in range(0, total, step)[:max_frames]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        return frames
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def _create_mosaic(frames: list[np.ndarray], cols: int = 4, target_width: int = 960) -> np.ndarray | None:
    """Compose frames into a grid mosaic."""
    if not frames:
        return None
    rows = (len(frames) + cols - 1) // cols
    h, w = frames[0].shape[:2]
    cell_w = target_width // cols
    cell_h = int(h * cell_w / w)
    canvas = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        resized = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        canvas[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = resized
    return canvas

def _bgr_to_pixmap(bgr: np.ndarray) -> QPixmap:
    arr = np.ascontiguousarray(bgr)
    h, w, ch = arr.shape
    qimg = QImage(arr.data, w, h, ch * w, QImage.Format.Format_BGR888).copy()
    return QPixmap.fromImage(qimg)
```

Display in QLabel:
```python
mosaic = _create_mosaic(frames, cols=4, target_width=960)
if mosaic is not None:
    mosaic_label = QLabel()
    pixmap = _bgr_to_pixmap(mosaic)
    mosaic_label.setPixmap(pixmap.scaled(960, 600, Qt.AspectRatioMode.KeepAspectRatio, ...))
    layout.addWidget(mosaic_label)
```

## SectionCard Component

Each VLM analysis segment is rendered as an independent card, containing: title + anomaly probability label + progress bar + mosaic keyframe image + Markdown analysis text. Uses native `QFrame.StyledPanel` style:

```python
class SectionCard(QFrame):
    def __init__(self, section: dict, index: int, total: int, parent=None):
        # section structure: {title, prob, start_frame, end_frame, analysis, clip_b64}
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        # ... build header, progress bar, VideoPlayer, QTextBrowser ...
```

**QTextBrowser Analysis Text Rendering**: Convert using the `markdown` library, then call `setHtml()`. After setting `setTextWidth`, retrieve the document height and use `setFixedHeight` for auto-sizing (paired with `ScrollBarAlwaysOff`).

## VLM Structured Return

`analyze_windows()` returns a `dict` instead of a `str`:

```python
{
    "markdown": str,       # Full Markdown report (for CLI/app.py usage)
    "sections": [          # For GUI card rendering
        {
            "title": "Segment 1 / 3",
            "prob": 0.87,
            "start_frame": 120,
            "end_frame": 184,
            "analysis": "VLM-returned Markdown analysis text",
            "clip_b64": "Base64-encoded MP4 video clip",
            "figure_number": 1,  # Figure number for Word export and VLM references
        },
        ...
    ],
    "summary": "Summary text",
}
```

### Word Report Export

After VLM analysis completes, display an "Export Report" button. On click, open a file save dialog and call `utils.export_report.export_vlm_report()` to generate a .docx file. Initialize `self._vlm_sections: list[dict] = []` and `self._vlm_summary: str = ""` in `__init__`, and cache the results in `_on_vlm_done`. See the `auto-skill-vlm-word-export` skill for details.

### _extract_clip_b64 Function

Extracts a specified frame range from the video and encodes it as a base64 MP4:

```python
def _extract_clip_b64(video_path, start_frame, end_frame,
                       resolution_h=240, target_fps=10) -> str | None:
    """
    1. Open video with cv2.VideoCapture
    2. Downsample frame indices by target_fps
    3. Write to temporary MP4 with cv2.VideoWriter (mp4v codec)
    4. Scale to resolution_h height (width proportional, keep even)
    5. Read file -> base64 encode -> delete temporary file
    """
```

**Resolution/FPS Selection**: 240p + 10fps keeps each clip under ~200KB (~270KB after base64 encoding). Three clips total ~800KB, avoiding GUI lag.

## UI Flow

```
User clicks "Submit Analysis"
       |
       v
  QStackedWidget switches to AnalysisPage
       |
       v
  InferenceWorker starts (QThread)
       |
       |-- Shows progress bar "Analyzing..."
       |
       v
  Inference complete -> Display statistics card + matplotlib chart
       |
       |-- Enable "VLM Deep Analysis" button
       |
       v
  User clicks VLM button -> VLMWorker starts
       |
       |-- _extract_clip_b64 extracts video clips
       |-- VLM analyzes each clip
       |
       v
  VLM complete -> SectionCard list rendered (with VideoPlayer)
       |
       v
  User clicks "<- Back to Recording" -> back_requested signal
       |
       v
  MainWindow._on_back_from_analysis() -> Switch back to _main_page
```

## Pitfalls and Gotchas

1. **`matplotlib.use('Agg')` must be called before importing pyplot**, otherwise the backend switch has no effect.
2. **FigureCanvas import path** is `matplotlib.backends.backend_qtagg` (PySide6/Qt6 uses `qtagg`; older versions use `qt5agg`).
3. **Clean up old charts**: Repeatedly adding Canvas widgets to the same layout causes stacking. You must call `deleteLater` on old widgets first.
4. **QThread abandon pattern**: After the page is destroyed, the worker may still be running and emitting signals. Use the `_abandoned` flag to prevent callbacks to a destroyed widget.
5. **QMediaPlayer does not support in-memory buffer playback**: You must write to a temporary file first, then load with `QUrl.fromLocalFile()`. Delete the file during cleanup.
6. **VLM return type change**: `analyze_windows()` changed from returning `str` to returning `dict`. All callers (app.py, CLI, GUI) must be updated accordingly.
7. **cv2.VideoWriter_fourcc Pylance error**: Add `# type: ignore[attr-defined]`.
8. **self.camera_thread is None Pylance error**: Add a guard `if self.camera_thread is None: return` before calling methods.
