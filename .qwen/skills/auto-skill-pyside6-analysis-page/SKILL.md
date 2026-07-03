---
name: pyside6-analysis-page
description: PySide6分析页架构：QStackedWidget页面切换 + QThread异步推理/VLM + matplotlib图表嵌入 + VideoPlayer卡片式VLM报告（含B64视频嵌入）
source: auto-skill
extracted_at: '2026-07-03T07:12:49.092Z'
---

# PySide6 分析页面架构

适用于"录制→回放→提交分析"流程中的分析阶段。回放结束后点击"提交分析"，通过 QStackedWidget 切换到分析页，自动运行推理并展示图表，再手动触发 VLM 深度分析（含视频片段嵌入的卡片式报告）。

## 整体架构

```
MainWindow (QStackedWidget)
├── _main_page  (录制/回放视图)
└── _analysis_page  (AnalysisPage widget)
     ├── QThread: InferenceWorker → 推理 → 图表 + 统计
     └── QThread: VLMWorker → VLM分析 → SectionCard 列表 (含 VideoPlayer)
```

## MainWindow 集成

### QStackedWidget 页面切换

```python
from PySide6.QtWidgets import QStackedWidget

self._stack = QStackedWidget()
self.setCentralWidget(self._stack)

self._main_page = QWidget()
main_layout = QVBoxLayout(self._main_page)
# ... 原有的录制/回放控件都加到 main_layout ...
self._stack.addWidget(self._main_page)

self._analysis_page: AnalysisPage | None = None
```

### 提交分析 → 切换页面

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

### closeEvent 清理

```python
def closeEvent(self, event):
    # ... 原有的 camera_thread / playback_thread 清理 ...
    if self._analysis_page:
        self._analysis_page.cleanup()
    super().closeEvent(event)
```

## AnalysisPage 组件

### QThread Worker 模式

推理和 VLM 调用都是耗时操作，必须放到后台线程。使用 `abandon` 标志防止页面销毁后回调：

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
        """页面销毁前调用，阻止 finished 信号发射"""
        self._abandoned = True

    def run(self):
        result = self._inference.predict(self._csv_path, self._output_json)
        if not self._abandoned:
            self.finished.emit(result)


class VLMWorker(QThread):
    finished = Signal(dict)  # 结构化结果，不是 str
    error = Signal(str)

    def run(self):
        try:
            result = analyze_windows(self.video_path, self.json_path, top_k=self.top_k)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
```

### 清理方法

```python
def cleanup(self):
    if self._inference_worker and self._inference_worker.isRunning():
        self._inference_worker.abandon()
        self._inference_worker.quit()
        self._inference_worker.wait(2000)
    if self._vlm_worker and self._vlm_worker.isRunning():
        self._vlm_worker.quit()
        self._vlm_worker.wait(5000)
    self._clear_sections()  # 清理 VideoPlayer 临时文件
```

### matplotlib 图表嵌入

用 `FigureCanvasQTAgg` 将 matplotlib 图表嵌入 Qt 布局。注意暗色主题适配：

```python
matplotlib.use('Agg')  # 必须在 import pyplot 前设置
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

def _plot_chart(self, r: dict):
    details = r.get("window_details", [])
    if not details:
        return

    # 清理旧图表（避免内存泄漏）
    for i in reversed(range(self._chart_layout.count())):
        w = self._chart_layout.itemAt(i).widget()
        if w:
            w.deleteLater()

    fig = Figure(figsize=(8, 3.5), dpi=100, facecolor='#1a1a2e')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#1a1a2e')

    x = [d["start_frame"] for d in details]
    y = [d["prob_abnormal"] for d in details]

    ax.plot(x, y, color='#e74c3c', marker='.', linewidth=1.2, markersize=4)
    ax.fill_between(x, y, alpha=0.15, color='#e74c3c')
    ax.set_xlabel("Frame", color='#aaa')
    ax.set_ylabel("P(abnormal)", color='#aaa')
    ax.set_title("逐窗口异常置信度", color='#eee', pad=10)
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors='#888')
    ax.grid(True, alpha=0.2, color='#555')
    for spine in ax.spines.values():
        spine.set_color('#444')

    fig.tight_layout()
    canvas = FigureCanvas(fig)
    canvas.setStyleSheet("background:transparent")
    self._chart_layout.addWidget(canvas)
```

## VideoPlayer 组件

基于 `QMediaPlayer` + `QVideoWidget` 的内嵌播放器，支持播放/暂停/进度条。视频数据从 base64 解码后写入临时 MP4 文件：

```python
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtCore import QUrl
import base64, tempfile, os

class VideoPlayer(QWidget):
    def __init__(self, clip_b64: str, parent=None):
        super().__init__(parent)
        self._tmp_file: str | None = None
        self._setup_ui()
        self._load_clip(clip_b64)

    def _load_clip(self, clip_b64: str):
        data = base64.b64decode(clip_b64)
        fd, path = tempfile.mkstemp(suffix='.mp4', dir=TEMP_DIR)
        os.write(fd, data)
        os.close(fd)
        self._tmp_file = path
        self._player.setSource(QUrl.fromLocalFile(path))

    def cleanup(self):
        self._player.stop()
        if self._tmp_file and os.path.exists(self._tmp_file):
            os.unlink(self._tmp_file)
```

**关键实现细节**：
- 使用 `QUrl.fromLocalFile()` 加载本地临时文件（QMediaPlayer 不支持直接从内存 buffer 播放 MP4）
- `QAudioOutput` 需要 `setMuted(True)` 静音（VLM 分析视频无音频）
- 进度条用 `sliderMoved` 信号实现拖动 seek，`positionChanged`/`durationChanged` 更新进度
- `cleanup()` 中必须删除临时文件，否则 TEMP_DIR 会堆积

## SectionCard 组件

每个 VLM 分析片段渲染为独立卡片，包含：标题 + 异常概率标签 + 进度条 + VideoPlayer + Markdown 分析文本。

```python
class SectionCard(QFrame):
    def __init__(self, section: dict, index: int, total: int, parent=None):
        # section 结构: {title, prob, start_frame, end_frame, analysis, clip_b64}
        # ... 构建 header、进度条、VideoPlayer、QTextBrowser ...
```

**QTextBrowser 渲染分析文本**：用 `markdown` 库转换，配合暗色主题 CSS。设置 `setTextWidth` 后获取文档高度，用 `setFixedHeight` 自适应高度（配合 `ScrollBarAlwaysOff`）。

## VLM 结构化返回

`analyze_windows()` 返回 `dict` 而非 `str`：

```python
{
    "markdown": str,       # 完整 Markdown 报告（供 CLI/app.py 使用）
    "sections": [          # 供 GUI 卡片渲染
        {
            "title": "问题片段 1 / 3",
            "prob": 0.87,
            "start_frame": 120,
            "end_frame": 184,
            "analysis": "VLM 返回的 Markdown 分析文本",
            "clip_b64": "base64 编码的 MP4 视频片段",
        },
        ...
    ],
    "summary": "总结文本",
}
```

### _extract_clip_b64 函数

从视频中截取指定帧范围，编码为 base64 MP4：

```python
def _extract_clip_b64(video_path, start_frame, end_frame,
                       resolution_h=240, target_fps=10) -> str | None:
    """
    1. cv2.VideoCapture 打开视频
    2. 按 target_fps 降采样帧索引
    3. cv2.VideoWriter 写入临时 MP4 (mp4v 编码器)
    4. 缩放到 resolution_h 高度（宽度按比例，保持偶数）
    5. 读取文件 → base64 编码 → 删除临时文件
    """
```

**分辨率/FPS 选择**：240p + 10fps 可将每个片段控制在 ~200KB 以内（base64 后 ~270KB），3 个片段共 ~800KB，避免 GUI 卡顿。

## UI 流程

```
用户点击"提交分析"
       │
       ▼
  QStackedWidget 切换到 AnalysisPage
       │
       ▼
  InferenceWorker 启动 (QThread)
       │
       ├── 显示进度条 "正在分析…"
       │
       ▼
  推理完成 → 显示统计卡片 + matplotlib 图表
       │
       ├── 启用 "🔍 VLM 深度分析" 按钮
       │
       ▼
  用户点击 VLM 按钮 → VLMWorker 启动
       │
       ├── _extract_clip_b64 截取视频片段
       ├── VLM 调用分析每个片段
       │
       ▼
  VLM 完成 → SectionCard 列表渲染（含 VideoPlayer）
       │
       ▼
  用户点击 "← 返回录制" → back_requested 信号
       │
       ▼
  MainWindow._on_back_from_analysis() → 切回 _main_page
```

## 踩坑点

1. **matplotlib.use('Agg') 必须在 import pyplot 之前**，否则后端切换无效。
2. **FigureCanvas 导入路径**是 `matplotlib.backends.backend_qtagg`（PySide6/Qt6 用 `qtagg`，旧版用 `qt5agg`）。
3. **清理旧图表**：往同一个 layout 里反复添加 Canvas 会叠加，必须先 `deleteLater` 旧 widget。
4. **QThread abandon 模式**：页面销毁后 worker 可能仍在运行并发信号，用 `_abandoned` 标志防止回调到已销毁的 widget。
5. **QMediaPlayer 不支持内存 buffer 播放**：必须写入临时文件后用 `QUrl.fromLocalFile()` 加载，cleanup 时删除。
6. **VLM 返回类型变更**：`analyze_windows()` 从返回 `str` 改为返回 `dict`，需同步更新所有调用方（app.py、CLI、GUI）。
7. **cv2.VideoWriter_fourcc Pylance 报错**：加 `# type: ignore[attr-defined]`。
8. **self.camera_thread is None Pylance 报错**：调用方法前加 `if self.camera_thread is None: return` 防护。
