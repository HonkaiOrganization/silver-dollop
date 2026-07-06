# Jump Rope Pose Recording & Analysis System

Real-time human pose capture and intelligent jump rope activity analysis powered by **YOLO11-pose**.

## Features

| Feature | Description |
|---------|-------------|
| Live Recording | Capture video via camera with real-time pose inference and recording |
| Skeleton Rendering | Render 17-keypoint human skeleton with joint angle annotations and grouped coloring |
| Playback | Replay recorded video and skeleton with pause and seek support |
| Sliding Window Inference | 1D-CNN based per-window anomaly detection on jump rope sequences |
| VLM Deep Analysis | Multimodal LLM analysis of anomalous segments with video-level report generation |
| Video Import | Import local video files to skip recording and proceed directly to pose inference |

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (GPU acceleration; CPU mode also supported)

### Installation

```bash
pip install -r requirements.txt
```

### Launch

```bash
python app.py
```

## Usage Guide

### 1. Camera Recording

1. After launching, the main window displays the camera feed on the left and skeleton rendering on the right
2. Select an available camera device from the **Camera** dropdown
3. Click **Start Recording** to begin; recording status updates in real time
4. Click **Stop Recording** when finished; the system saves automatically and enters playback mode

> **Note**: Recordings must be at least 10 seconds; shorter recordings are discarded.

### 2. Playback

Playback mode starts automatically after recording:

- **Play / Pause**: Control playback
- **Progress slider**: Seek to any frame
- **Re-record**: Discard current recording and return to camera mode
- **Submit for Analysis**: Proceed to the AI analysis pipeline

### 3. Import External Video

Use **File -> Open Video File** (or `Ctrl+O`) to import a local video:

- Supported formats: `.mp4` `.avi` `.mkv` `.mov` `.wmv` `.flv`
- The system performs frame-by-frame pose inference to generate skeleton data and video
- After import, playback mode starts automatically; you can submit for analysis directly
- Progress is shown in a dialog; import can be cancelled

### 4. AI Analysis Pipeline

Click **Submit for Analysis** to enter the analysis page:

1. **Sliding Window Inference**: Automatically classifies keypoint data using 1D-CNN, outputting prediction (Normal/Abnormal), confidence, probability distribution, and summary statistics
2. **Abnormal Probability Chart**: Line chart showing per-window abnormal confidence trend
3. **VLM Deep Analysis** (optional): Click **VLM Deep Analysis** to extract the top-3 most anomalous windows, invoke a multimodal LLM for analysis, and generate detailed card-style reports (with video playback + textual analysis)
4. **Back**: Click the top-left button to return to the main interface

## Menu Bar

| Menu | Action | Shortcut |
|------|--------|----------|
| File -> Open Video File | Open a local video for direct analysis | `Ctrl+O` |
| File -> Exit | Close the application | `Ctrl+Q` |
| Help -> View Help Documentation | Open this README | `F1` |
| Help -> About | Show application info | -- |

## Project Structure

```
sport/
├── app.py                  # PySide6 desktop application entry point
├── config.py               # Global configuration and keypoint definitions
├── core/
│   ├── extractor/          # Pose keypoint extraction (PoseExtractor)
│   ├── infer/              # Sliding window classification inference (JumpRopeInference)
│   ├── visualizer/         # Skeleton visualization
│   └── vlm/                # VLM multimodal analysis
├── gui/
│   ├── main_window.py      # Main window
│   ├── analysis_page.py    # Analysis page (inference + VLM + charts)
│   ├── camera_thread.py    # Camera recording thread
│   ├── playback_thread.py  # Playback thread
│   ├── file_import_thread.py  # Video file import thread
│   └── frame_processor.py  # Frame cropping (9:16)
├── models/
│   ├── camera/             # Camera management (CameraManager)
│   └── pose/               # Pose inference (PoseProcessor, YOLO11-pose)
├── pretrained/             # Pretrained model files
├── temp/                   # Temporary recording/import files
├── output/                 # Output files
└── utils/                  # Utility functions
```

## Tech Stack

- **Pose Estimation**: YOLO11m-pose (Ultralytics)
- **Classification Model**: 1D-CNN sliding window classifier
- **GUI Framework**: PySide6 (Qt 6)
- **Video Processing**: OpenCV
- **Data Format**: CSV (17 keypoints x 3 channels: x, y, conf)
- **VLM Analysis**: Alibaba Cloud DashScope Multimodal API
- **Charts**: matplotlib (embedded in Qt interface)

## FAQ

### Q: Camera not detected?

A: Ensure the camera driver is properly installed. On Windows, the DirectShow backend is used; check camera status in Device Manager.

### Q: Slow inference speed?

A: Ensure CUDA-enabled PyTorch is installed; the model will automatically use GPU inference. CPU mode is significantly slower.

### Q: Import progress stuck?

A: Longer videos require more processing time. Use the Cancel button in the progress dialog to interrupt the import.

### Q: VLM analysis not working?

A: VLM analysis requires an Alibaba Cloud DashScope API key. Ensure the environment variable or configuration is set correctly.
