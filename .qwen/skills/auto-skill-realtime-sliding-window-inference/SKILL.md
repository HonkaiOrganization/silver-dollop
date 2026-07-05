---
name: realtime-sliding-window-inference
description: 实时滑动窗口分类推理：QThread累积关键点(置信度过滤)+首帧归一化+滑窗推理+Abnormal概率实时显示，录制/回放双模式共用
source: auto-skill
extracted_at: '2026-07-05T03:36:18.247Z'
---

# 实时滑动窗口分类推理

在摄像机录制/回放页面集成实时滑窗分类器，边采集关键点边推理，表格下方显示异常概率。

## 整体架构

```
CameraThread / PlaybackThread
        │
        │ keypoints_ready(xy: [17,2], conf: [17])
        ▼
RealtimeInferenceThread (QThread)
  ├── add_keypoints(xy, conf)
  │     └── 按 mean(conf) >= 0.5 过滤垃圾帧
  ├── _buffer: list[ndarray]        ← 仅累积高置信度帧
  ├── 首帧归一化（髋中心 + 肩宽，肩宽<1.0则跳过）
  ├── 每帧检查 len(buffer) >= window_size
  └── 取最新 window_size 帧 → 模型推理
        │
        │ result_ready(float)  — P(abnormal), NaN=尚无结果
        ▼
MainWindow._on_inference_result()
  └── lbl_confidence.setText("Abnormal: 32.50%")
```

## 关键坑点：必须按置信度过滤

**问题**：YOLO 即使没检测到人也会返回低置信度的 xy 坐标。如果不过滤：
- 垃圾帧污染缓冲区，归一化参考帧可能是无效关键点
- 肩宽计算错误 → 所有归一化坐标爆炸 → 模型输出极端概率 → 永远显示 100%

**解决**：`keypoints_ready` 信号携带 `(xy, conf)`，推理线程按 `mean(conf) >= 0.5` 过滤：

```python
def add_keypoints(self, xy, conf):
    if xy is None or conf is None:
        return
    mean_conf = float(np.mean(conf))
    if mean_conf < self._CONF_THRESH:  # 0.5
        return
    self._buffer.append(xy.copy())
```

## RealtimeInferenceThread 实现

### 类定义

```python
class RealtimeInferenceThread(QThread):
    result_ready = Signal(float)  # P(abnormal), NaN 表示尚无结果
    _CONF_THRESH = 0.5

    def __init__(self, model_path="pretrained/model_export.pt"):
        super().__init__()
        self.model_path = model_path
        self._buffer: list[np.ndarray] = []
        self._ref_hip: np.ndarray | None = None
        self._ref_shoulder_width: float | None = None
        self._model = None
        self._window_size = 0
        self._stride = 1
        self._device = None

    def reset(self):
        """模式切换时清空缓冲区"""
        self._buffer.clear()
        self._ref_hip = None
        self._ref_shoulder_width = None
        self.result_ready.emit(float("nan"))
```

### 线程主体：模型加载 + 推理循环

模型在 `run()` 中加载（在 worker 线程内），避免阻塞主线程：

```python
def run(self):
    self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(self.model_path):
        return  # 模型不存在，静默退出

    export_data = torch.load(self.model_path, map_location=self._device, weights_only=True)
    model_cfg = export_data["config"]
    self._model = JumpRopeClassifier(...).to(self._device)
    self._model.load_state_dict(export_data["model_state_dict"])
    self._model.eval()
    self._window_size = model_cfg["window_size"]
    self._stride = model_cfg.get("stride", 32)

    last_infer_len = 0
    while self._is_running:
        buf_len = len(self._buffer)
        if buf_len >= self._window_size and buf_len != last_infer_len:
            self._run_inference()
            last_infer_len = buf_len
        self.msleep(30)
```

### 归一化 + 推理

用**首帧有效检测**的髋部中点和肩宽作为固定参考。肩宽 < 1.0 像素视为异常，跳过本次推理：

```python
def _run_inference(self):
    coords = np.array(self._buffer)[:, :, :2]  # (N, 17, 2)

    if self._ref_hip is None:
        mid_hip = (coords[0, 11] + coords[0, 12]) / 2.0
        shoulder_vec = coords[0, 6] - coords[0, 5]
        sw = float(np.linalg.norm(shoulder_vec))
        if sw < 1.0:
            return  # 肩宽异常，等下一帧
        self._ref_hip = mid_hip
        self._ref_shoulder_width = sw

    coords = coords - self._ref_hip[np.newaxis, np.newaxis, :]
    coords = coords / self._ref_shoulder_width

    start = max(0, len(coords) - self._window_size)
    window = coords[start:start + self._window_size]
    if len(window) < self._window_size:
        self.result_ready.emit(float("nan"))
        return

    window_flat = window.reshape(self._window_size, -1)
    input_tensor = torch.tensor(window_flat, dtype=torch.float32).unsqueeze(0).to(self._device)

    with torch.no_grad():
        logits = self._model(input_tensor)
        probs = F.softmax(logits, dim=1)
        prob_abnormal = float(probs[0, 0].cpu())

    self.result_ready.emit(prob_abnormal)
```

## MainWindow 集成

### 布局：表格 + 置信度面板

置信度标签放在表格正下方，用垂直面板包裹：

```python
kpt_panel = QWidget()
kpt_panel.setFixedWidth(280)
kpt_panel_layout = QVBoxLayout(kpt_panel)

self.kpt_table = QTableWidget(17, 3)  # 关键点表格
kpt_panel_layout.addWidget(self.kpt_table, 1)

self.lbl_confidence = QLabel("Abnormal: NaN")
self.lbl_confidence.setAlignment(Qt.AlignmentFlag.AlignCenter)
self.lbl_confidence.setStyleSheet(
    "font-size: 15px; font-weight: bold;"
    "background: rgba(0,0,0,180); color: white;"
    "padding: 5px 10px; border-radius: 4px;"
)
kpt_panel_layout.addWidget(self.lbl_confidence)

views_layout.addWidget(kpt_panel)  # 作为右侧第三列
```

### 初始化顺序

推理线程必须在 `init_ui()` 之后创建（需要 `lbl_confidence` 存在），在 `init_camera_system()` 之前创建（camera 连接时需要目标存在）：

```python
def __init__(self):
    ...
    self.init_ui()
    self.init_inference_thread()   # ← 先创建推理线程
    self.init_camera_system()      # ← 再连接信号

def init_inference_thread(self):
    self._inference_thread = RealtimeInferenceThread()
    self._inference_thread.result_ready.connect(self._on_inference_result)
    self._inference_thread.start()
```

### 信号连接（录制 + 回放共用）

CameraThread 和 PlaybackThread 的 `keypoints_ready(object, object)` 发射 `(xy, conf)`，同时连接两个 slot：

```python
# init_camera_system 中
self.camera_thread.keypoints_ready.connect(self._on_keypoints_updated)           # 更新表格
self.camera_thread.keypoints_ready.connect(self._inference_thread.add_keypoints)  # 累积推理

# _switch_to_playback 中
self.playback_thread.keypoints_ready.connect(self._on_keypoints_updated)
self.playback_thread.keypoints_ready.connect(self._inference_thread.add_keypoints)
```

### 显示与重置

```python
def _on_inference_result(self, confidence: float):
    if math.isnan(confidence):
        self.lbl_confidence.setText("Abnormal: NaN")
    else:
        self.lbl_confidence.setText(f"Abnormal: {confidence:.2%}")

# 模式切换时重置
self._inference_thread.reset()
self.lbl_confidence.setText("Abnormal: NaN")
```

### closeEvent 清理

```python
if self._inference_thread:
    self._inference_thread.stop()
```

## 关键设计决策

| 决策 | 理由 |
|------|------|
| **按 mean(conf) 过滤垃圾帧** | YOLO 无人时也返回低置信度 xy，不过滤会导致归一化崩溃、模型输出 100% |
| 首帧归一化（固定参考） | 实时场景无法预知全部帧，用首帧有效检测的 hip/shoulder 作为稳定参考点 |
| 肩宽 < 1.0 跳过推理 | 首帧若检测到极小人体，归一化坐标会爆炸 |
| 每帧都推理（非仅 stride 整数倍） | 实时反馈需要尽可能频繁更新，最新 window_size 帧滑动 |
| 模型在 worker 线程加载 | 避免阻塞主线程 UI（加载 PT 模型可能耗时数秒） |
| `add_keypoints` 通过 Qt 信号队列调用 | 线程安全：发射线程(MainThread) → 排队 → worker 线程执行 |
| 显示 P(abnormal) 而非 P(normal) | 用户关注异常概率，normal 概率高不直观 |
| 模型不存在时静默退出 | 开发阶段可能还没有训练好的模型，不应崩溃 |
