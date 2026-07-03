---
name: pyside6-analysis-page
description: PySide6分析页架构：QStackedWidget页面切换 + QThread异步推理/VLM + matplotlib图表嵌入 + QTextBrowser Markdown渲染
source: auto-skill
extracted_at: '2026-07-03T05:46:14.120Z'
---

# PySide6 分析页面架构

适用于"录制→回放→提交分析"流程中的分析阶段。回放结束后点击"提交分析"，通过 QStackedWidget 切换到分析页，自动运行推理并展示图表，再手动触发 VLM 深度分析。

## 整体架构

```
MainWindow (QStackedWidget)
├── _main_page  (录制/回放视图)
└── _analysis_page  (AnalysisPage widget)
     ├── QThread: InferenceWorker → 推理 → 图表 + 统计
     └── QThread: VLMWorker → VLM分析 → Markdown 报告渲染
```

## MainWindow 集成

### QStackedWidget 页面切换

将原来的 `setCentralWidget(main_widget)` 改为 QStackedWidget 容器：

```python
from PySide6.QtWidgets import QStackedWidget

# __init__ 中
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

    # 清理旧的分析页面
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
    finished = Signal(str)
    error = Signal(str)

    def run(self):
        try:
            report = analyze_windows(self.video_path, self.json_path, top_k=self.top_k)
            self.finished.emit(report)
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
```

### matplotlib 图表嵌入

用 `FigureCanvasQTAgg` 将 matplotlib 图表嵌入 Qt 布局。注意暗色主题适配：

```python
matplotlib.use('Agg')  # 必须在 import pyplot 前设置
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
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

**注意**：`FigureCanvasQTAgg` 的导入路径是 `matplotlib.backends.backend_qt5agg`（即使使用 PySide6/Qt6，matplotlib 的 Qt5 backend 仍然兼容）。

### QTextBrowser Markdown 渲染

QTextBrowser 只支持 HTML 子集，需要将 Markdown 转为 HTML。轻量级实现（无需外部 `markdown` 库）：

```python
import re

def _md_to_html(md: str) -> str:
    lines = md.split('\n')
    html_lines = []
    in_code_block = False
    in_list = False

    for line in lines:
        stripped = line.strip()

        # 代码块
        if stripped.startswith('```'):
            if in_code_block:
                html_lines.append('</code></pre>')
                in_code_block = False
            else:
                html_lines.append('<pre style="background:#2b2b2b;padding:8px;border-radius:4px"><code>')
                in_code_block = True
            continue

        if in_code_block:
            html_lines.append(line.replace('<', '&lt;').replace('>', '&gt;'))
            continue

        # 空行
        if not stripped:
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            continue

        # 标题
        m = re.match(r'^(#{1,6})\s+(.+)', stripped)
        if m:
            html_lines.append(f'<h{len(m.group(1))}>{_inline_md(m.group(2))}</h{len(m.group(1))}>')
            continue

        # 无序列表
        if stripped.startswith('- ') or stripped.startswith('* '):
            if not in_list:
                html_lines.append('<ul>')
                in_list = True
            html_lines.append(f'<li>{_inline_md(stripped[2:])}</li>')
            continue

        if in_list:
            html_lines.append('</ul>')
            in_list = False

        # 分割线
        if stripped == '---':
            html_lines.append('<hr>')
            continue

        html_lines.append(f'<p>{_inline_md(stripped)}</p>')

    if in_list:
        html_lines.append('</ul>')
    return '\n'.join(html_lines)

def _inline_md(text: str) -> str:
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`(.+?)`', r'<code style="background:#2b2b2b;padding:1px 4px;border-radius:3px">\1</code>', text)
    return text
```

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
       ├── 显示进度条 "VLM 分析中…"
       │
       ▼
  VLM 完成 → QTextBrowser 渲染 Markdown 报告
       │
       ▼
  用户点击 "← 返回录制" → back_requested 信号
       │
       ▼
  MainWindow._on_back_from_analysis() → 切回 _main_page
```

## 踩坑点

1. **matplotlib.use('Agg') 必须在 import pyplot 之前**，否则后端切换无效。
2. **FigureCanvas 导入路径**是 `matplotlib.backends.backend_qt5agg`，即使使用 PySide6 也是如此——这是 matplotlib 的 Qt5 兼容层。
3. **清理旧图表**：往同一个 layout 里反复添加 Canvas 会叠加，必须先 `deleteLater` 旧 widget。
4. **QThread abandon 模式**：页面销毁后 worker 可能仍在运行并发信号，用 `_abandoned` 标志防止回调到已销毁的 widget。
5. **`cv2.VideoWriter_fourcc` Pylance 报错**：加 `# type: ignore[attr-defined]` 即可，这是 OpenCV C 扩展的动态属性，stub 不认识。
6. **`self.camera_thread is None` Pylance 报错**：初始化时为 `None`，调用方法前加 `if self.camera_thread is None: return` 防护检查。
