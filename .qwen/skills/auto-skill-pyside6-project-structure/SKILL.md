---
name: pyside6-project-structure
description: PySide6 GUI项目工程化规范：包结构拆分、公共常量提取、widgets/workers分层、原生风格UI、logging替代print
source: auto-skill
extracted_at: '2026-07-04T12:00:00.000Z'
---

# PySide6 项目工程化规范

将"能跑就行"的 PySide6 项目重构为符合通用 GUI 项目规范的工程化结构。

## 目标目录结构

```
project_root/
├── app.py                    # 入口：QApplication + MainWindow
├── config.py                 # 全局配置 + 公共常量 + logging 配置
├── requirements.txt          # 依赖清单
├── core/                     # 业务逻辑层（与 GUI 无关）
│   ├── __init__.py           # 空文件（标记为包）
│   ├── extractor/
│   │   ├── __init__.py       # 纯导出：from .extractor import PoseExtractor
│   │   └── extractor.py      # 实际实现
│   ├── infer/
│   │   ├── __init__.py
│   │   └── inference.py
│   └── ...
├── models/                   # 数据/模型层
│   ├── __init__.py
│   ├── camera/
│   │   ├── __init__.py
│   │   └── manager.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── classifier.py
│   └── pose/
│       ├── __init__.py
│       └── processor.py
├── gui/                      # GUI 层
│   ├── __init__.py
│   ├── main_window.py        # 主窗口
│   ├── analysis_page.py      # 分析页（仅页面逻辑）
│   ├── camera_thread.py      # 线程类
│   ├── playback_thread.py
│   ├── file_import_thread.py
│   ├── frame_processor.py
│   ├── widgets/              # 可复用 UI 组件
│   │   ├── __init__.py
│   │   ├── video_player.py
│   │   └── section_card.py
│   └── workers/              # 后台线程（QThread）
│       ├── __init__.py
│       ├── inference_worker.py
│       └── vlm_worker.py
├── utils/                    # 工具函数
│   ├── __init__.py
│   └── load_csv/
│       ├── __init__.py
│       └── loader.py
└── legacy/                   # 旧版代码（不删除，归档）
    └── gui.py
```

## 核心原则

### 1. `__init__.py` 只做导出

实现代码永远不要放在 `__init__.py` 里。`__init__.py` 只做一件事：从同目录的模块文件 re-export。

```python
# core/extractor/__init__.py
from .extractor import PoseExtractor
```

**好处**：避免循环导入、IDE 跳转更准确、模块可独立测试。

### 2. 公共常量集中管理

`KPT_NAMES`、`CSV_COLUMNS`、`SKELETON_LINKS` 等常量如果出现在 2+ 个文件中，必须提取到 `config.py`。

```python
# config.py
KPT_NAMES = ["nose", "L_eye", ...]
CSV_COLUMNS = ["frame_id", "person_id"] + [...]
SKELETON_LINKS = [(15, 13), (13, 11), ...]
```

所有引用方：`from config import KPT_NAMES, CSV_COLUMNS`

### 3. GUI 分层：widgets/ + workers/

当一个页面文件超过 200 行或包含 3+ 个类时，拆分：

- `gui/widgets/` — 可复用 UI 组件（VideoPlayer、SectionCard 等）
- `gui/workers/` — QThread 后台线程（InferenceWorker、VLMWorker 等）
- 页面文件只保留页面级布局和事件处理

### 4. 原生 PySide 风格

**不要**大量使用 `setStyleSheet` 硬编码暗色主题。保持系统原生外观：

```python
# 好的做法：仅保留功能性样式
self.camera_view.setStyleSheet("background-color: black;")  # 视频区域需要黑底

# 不好的做法：自定义按钮颜色、圆角、渐变
self.btn.setStyleSheet(
    "QPushButton{background:#c0392b;color:#fff;border-radius:4px;padding:6px 18px}"
    "QPushButton:hover{background:#e74c3c}"
)
```

**替代方案**：
- 用 `QFrame.Shape.StyledPanel` 代替自定义卡片样式
- 用 `QStyle.StandardPixmap` 标准图标
- 按钮使用系统默认外观
- matplotlib 图表使用默认配色（不设置 facecolor）

### 5. logging 替代 print

```python
# config.py 中统一配置
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

# 各模块中使用
logger = logging.getLogger(__name__)
logger.info("录制完成，保存视频: %s, CSV: %s", video_path, csv_path)
```

### 6. 包级 `__init__.py` 不可省略

`core/`、`models/`、`utils/` 等顶层包目录必须有 `__init__.py`（可以是空文件），否则 Python 不会识别为包。

### 7. requirements.txt

```
PySide6>=6.5
ultralytics>=8.0
opencv-python>=4.8
torch>=2.0
numpy
pandas
matplotlib
dashscope
markdown
requests
click
```

### 8. 旧代码归档而非删除

旧版 `gui.py`（Gradio WebUI）移入 `legacy/` 目录，保留可追溯性。

## 重构检查清单

- [ ] 所有 `__init__.py` 中的实现代码移到独立 `.py` 文件
- [ ] 重复常量提取到 `config.py`
- [ ] 超过 200 行的 GUI 文件拆分为 widgets/ + workers/
- [ ] 移除硬编码暗色 `setStyleSheet`，保留功能性样式
- [ ] `print()` → `logger.info/debug/error()`
- [ ] 添加 `requirements.txt`
- [ ] 所有包目录有 `__init__.py`
- [ ] 旧代码移入 `legacy/`
- [ ] 验证所有导入：`python -c "from gui.main_window import MainWindow"`
