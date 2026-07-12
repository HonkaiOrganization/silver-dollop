---
name: pyside6-project-structure
description: PySide6 GUI project engineering standards — package structure decomposition, shared constant extraction, widgets/workers layering, native-style UI, logging over print
source: auto-skill
extracted_at: '2026-07-04T12:00:00.000Z'
---

# PySide6 Project Engineering Standards

Restructure a "just-make-it-run" PySide6 project into an engineering-grade layout that follows standard GUI project conventions.

## Target Directory Structure

```
project_root/
├── app.py                    # Entry point: QApplication + MainWindow
├── config.py                 # Global configuration + shared constants + logging setup
├── requirements.txt          # Dependency manifest
├── core/                     # Business logic layer (GUI-independent)
│   ├── __init__.py           # Empty file (marks directory as a package)
│   ├── infer/
│   │   ├── __init__.py
│   │   └── inference.py
│   ├── vlm/
│   │   ├── __init__.py
│   │   └── analyzer.py
│   └── export_report/
│       └── __init__.py       # Word report export (module, not utility)
├── models/                   # Data / model layer (single source of truth for pose)
│   ├── __init__.py
│   ├── camera/
│   │   ├── __init__.py
│   │   └── manager.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── classifier.py
│   └── pose/
│       ├── __init__.py
│       └── processor.py      # PoseProcessor: YOLO inference + skeleton rendering
├── gui/                      # GUI layer
│   ├── __init__.py
│   ├── main_window.py        # Main window
│   ├── analysis_page.py      # Analysis page (page-level logic only)
│   ├── camera_thread.py      # Thread classes
│   ├── playback_thread.py
│   ├── file_import_thread.py
│   ├── frame_processor.py
│   ├── widgets/              # Reusable UI components
│   │   ├── __init__.py
│   │   ├── frame_display.py  # FrameDisplay base (BGR→QPixmap + resize)
│   │   ├── video_display.py  # VideoDisplay + SkeletonDisplay (subclasses)
│   │   └── section_card.py   # SectionCard (mosaic image, no video player)
│   └── workers/              # Background threads (QThread)
│       ├── __init__.py
│       ├── inference_worker.py
│       └── vlm_worker.py
├── utils/                    # Utility functions
│   ├── __init__.py
│   └── load_csv/
│       ├── __init__.py
│       └── loader.py
└── legacy/                   # Legacy code (archived, not deleted)
    └── app_gradio.py
```

**Key principle**: Do NOT duplicate YOLO pose logic across packages. `models/pose/processor.py` is the single source of truth for pose estimation and skeleton rendering. Both recording (CameraThread) and playback (PlaybackThread) use `PoseProcessor`.

## Core Principles

### 1. `__init__.py` for Re-exports Only

Implementation code must never reside in `__init__.py`. The sole purpose of `__init__.py` is to re-export symbols from sibling module files within the same directory.

```python
# core/extractor/__init__.py
from .extractor import PoseExtractor
```

**Benefits**: Avoids circular imports, improves IDE jump-to-definition accuracy, and enables independent module testing.

### 2. Centralize Shared Constants

Constants such as `KPT_NAMES`, `CSV_COLUMNS`, `SKELETON_LINKS` must be extracted to `config.py` if they appear in 2 or more files.

```python
# config.py
KPT_NAMES = ["nose", "L_eye", ...]
CSV_COLUMNS = ["frame_id", "person_id"] + [...]
SKELETON_LINKS = [(15, 13), (13, 11), ...]
```

All consumers: `from config import KPT_NAMES, CSV_COLUMNS`

### 3. GUI Layering: widgets/ + workers/

When a single page file exceeds 200 lines or contains 3 or more classes, decompose it:

- `gui/widgets/` — Reusable UI components (VideoPlayer, SectionCard, etc.)
- `gui/workers/` — QThread background workers (InferenceWorker, VLMWorker, etc.)
- The page file retains only page-level layout and event handling

### 4. Native PySide Style

**Do not** use extensive `setStyleSheet` calls to hardcode a dark theme. Preserve the native system appearance:

```python
# Good practice: functional styles only
self.camera_view.setStyleSheet("background-color: black;")  # Video area requires a black background

# Bad practice: custom button colors, rounded corners, gradients
self.btn.setStyleSheet(
    "QPushButton{background:#c0392b;color:#fff;border-radius:4px;padding:6px 18px}"
    "QPushButton:hover{background:#e74c3c}"
)
```

**Alternatives**:
- Use `QFrame.Shape.StyledPanel` instead of custom card styles
- Use `QStyle.StandardPixmap` for standard icons
- Keep buttons with the default system appearance
- Use default color schemes for matplotlib charts (do not set `facecolor`)

### 5. Use logging Instead of print

```python
# Unified configuration in config.py
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

# In each module
logger = logging.getLogger(__name__)
logger.info("Recording complete — saved video: %s, CSV: %s", video_path, csv_path)
```

### 6. Package-Level `__init__.py` Is Mandatory

Top-level package directories such as `core/`, `models/`, `utils/` must each contain an `__init__.py` file (it may be empty); otherwise Python will not recognize the directory as a package.

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

### 8. Archive Legacy Code Instead of Deleting It

Move the legacy `gui.py` (Gradio WebUI) into the `legacy/` directory to preserve traceability.

## Refactoring Checklist

- [ ] Move all implementation code out of `__init__.py` files into dedicated `.py` modules
- [ ] Extract duplicate constants into `config.py`
- [ ] Split GUI files exceeding 200 lines into widgets/ + workers/
- [ ] Remove hardcoded dark-theme `setStyleSheet` calls; retain functional styles only
- [ ] Replace `print()` with `logger.info/debug/error()`
- [ ] Add `requirements.txt`
- [ ] Ensure every package directory contains an `__init__.py`
- [ ] Move legacy code into `legacy/`
- [ ] Verify all imports: `python -c "from gui.main_window import MainWindow"`
