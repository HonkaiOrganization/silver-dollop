import os
import logging
import math
import numpy as np
import torch
import torch.nn.functional as F
from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


class RealtimeInferenceThread(QThread):
    """
    实时滑动窗口推理线程。
    接收原始关键点 (xy, conf)，按置信度过滤后累积，
    用首帧有效检测做空间归一化 + 滑窗分类。
    """
    result_ready = Signal(float)  # P(abnormal), NaN 表示尚无结果

    _CONF_THRESH = 0.5

    def __init__(self, model_path: str = "pretrained/model_export.pt"):
        super().__init__()
        self.model_path = model_path
        self._buffer: list[np.ndarray] = []
        self._ref_hip: np.ndarray | None = None
        self._ref_shoulder_width: float | None = None
        self._is_running = False
        self._model = None
        self._window_size = 0
        self._stride = 1
        self._device = None

    def reset(self):
        self._buffer.clear()
        self._ref_hip = None
        self._ref_shoulder_width = None
        self.result_ready.emit(float("nan"))

    def add_keypoints(self, xy, conf):
        if xy is None or conf is None:
            return
        mean_conf = float(np.mean(conf))
        if mean_conf < self._CONF_THRESH:
            return
        self._buffer.append(xy.copy())

    def run(self):
        self._is_running = True
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(self.model_path):
            logger.warning("推理模型不存在: %s", self.model_path)
            return

        try:
            export_data = torch.load(self.model_path, map_location=self._device,
                                     weights_only=True)
            model_cfg = export_data["config"]
            from models.model import JumpRopeClassifier
            self._model = JumpRopeClassifier(
                in_channels=model_cfg["in_channels"],
                num_classes=model_cfg["num_classes"],
            ).to(self._device)
            self._model.load_state_dict(export_data["model_state_dict"])
            self._model.eval()
            self._window_size = model_cfg["window_size"]
            self._stride = model_cfg.get("stride", 32)
            logger.info("实时推理线程已加载模型 (window=%d, stride=%d)",
                        self._window_size, self._stride)
        except Exception:
            logger.exception("加载推理模型失败")
            return

        last_infer_len = 0
        while self._is_running:
            buf_len = len(self._buffer)
            if buf_len >= self._window_size and buf_len != last_infer_len:
                self._run_inference()
                last_infer_len = buf_len
            self.msleep(30)

    def stop(self):
        self._is_running = False
        self.wait(3000)

    def _run_inference(self):
        try:
            coords = np.array(self._buffer)[:, :, :2]

            if self._ref_hip is None:
                mid_hip = (coords[0, 11] + coords[0, 12]) / 2.0
                shoulder_vec = coords[0, 6] - coords[0, 5]
                sw = float(np.linalg.norm(shoulder_vec))
                if sw < 1.0:
                    logger.warning("肩宽异常 (%.2f)，跳过本次推理", sw)
                    return
                self._ref_hip = mid_hip
                self._ref_shoulder_width = sw
                logger.info("归一化参考: hip=(%.1f, %.1f), shoulder_width=%.1f",
                            mid_hip[0], mid_hip[1], sw)

            coords = coords - self._ref_hip[np.newaxis, np.newaxis, :]
            coords = coords / self._ref_shoulder_width

            start = max(0, len(coords) - self._window_size)
            window = coords[start:start + self._window_size]
            if len(window) < self._window_size:
                self.result_ready.emit(float("nan"))
                return

            window_flat = window.reshape(self._window_size, -1)
            input_tensor = torch.tensor(
                window_flat, dtype=torch.float32
            ).unsqueeze(0).to(self._device)

            with torch.no_grad():
                logits = self._model(input_tensor)
                probs = F.softmax(logits, dim=1)
                prob_abnormal = float(probs[0, 0].cpu())
                prob_normal = float(probs[0, 1].cpu())

            logger.debug("推理结果: P(normal)=%.4f P(abnormal)=%.4f buf=%d",
                         prob_normal, prob_abnormal, len(self._buffer))

            result = prob_abnormal
            if math.isnan(result) or math.isinf(result):
                result = float("nan")
            self.result_ready.emit(result)
        except Exception:
            logger.exception("实时推理异常")
            self.result_ready.emit(float("nan"))
