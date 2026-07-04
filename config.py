"""
全局配置模块。
通过环境变量 POSE_ENGINE 控制姿态引擎选择。
支持的取值：
  - "YOLO"       (默认) 使用 YOLO-Pose
  - "MediaPipe"  使用 Google MediaPipe Pose
"""
import os
import logging

# ── 日志配置 ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ── 公共常量：YOLO COCO 17 关键点 ────────────────────────────────────────
KPT_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_sho", "R_sho", "L_elb", "R_elb",
    "L_wri", "R_wri", "L_hip", "R_hip",
    "L_kne", "R_kne", "L_ank", "R_ank",
]

CSV_COLUMNS = ["frame_id", "person_id"] + [
    f"{name}_{suffix}" for name in KPT_NAMES for suffix in ("x", "y", "conf")
]

SKELETON_LINKS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (0, 1), (1, 3), (0, 2), (2, 4),
]

# ── 姿态引擎 ────────────────────────────────────────────────────────────
POSE_ENGINE: str = os.getenv("POSE_ENGINE", "YOLO").strip()
assert POSE_ENGINE in ("YOLO", "MediaPipe"), (
    f"POSE_ENGINE 取值无效: {POSE_ENGINE!r}，仅支持 'YOLO' 或 'MediaPipe'"
)

# ── MediaPipe Tasks API 模型路径 ─────────────────────────────────────────
# Tasks API 需要 .task 模型文件，首次使用时自动下载
# 可选模型:
#   pose_landmarker_lite  — 最快，精度较低 (~5MB)
#   pose_landmarker_full  — 均衡，精度较高 (~30MB)  ← 默认
#   pose_landmarker_heavy — 最慢，精度最高 (~50MB)
MEDIAPIPE_MODEL_PATH: str = os.getenv(
    "MEDIAPIPE_MODEL_PATH",
    "pretrained/pose_landmarker_full.task",
).strip()

MEDIAPIPE_MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)

# ── MediaPipe ↔ YOLO 关键点映射 ─────────────────────────────────────────
# YOLO COCO 17 点名称（与 KPT_NAMES 保持一致）
YOLO_KPT_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_sho", "R_sho", "L_elb", "R_elb",
    "L_wri", "R_wri", "L_hip", "R_hip",
    "L_kne", "R_kne", "L_ank", "R_ank",
]

# MediaPipe 33 个原始关键点名称
MEDIAPIPE_RAW_NAMES = [
    "nose",                          # 0
    "left_eye_inner",                # 1
    "left_eye",                      # 2
    "left_eye_outer",                # 3
    "right_eye_inner",               # 4
    "right_eye",                     # 5
    "right_eye_outer",               # 6
    "left_ear",                      # 7
    "right_ear",                     # 8
    "mouth_left",                    # 9
    "mouth_right",                   # 10
    "left_shoulder",                 # 11
    "right_shoulder",                # 12
    "left_elbow",                    # 13
    "right_elbow",                   # 14
    "left_wrist",                    # 15
    "right_wrist",                   # 16
    "left_pinky",                    # 17
    "right_pinky",                   # 18
    "left_index",                    # 19
    "right_index",                   # 20
    "left_thumb",                    # 21
    "right_thumb",                   # 22
    "left_hip",                      # 23
    "right_hip",                     # 24
    "left_knee",                     # 25
    "right_knee",                    # 26
    "left_ankle",                    # 27
    "right_ankle",                   # 28
    "left_heel",                     # 29
    "right_heel",                    # 30
    "left_foot_index",               # 31
    "right_foot_index",              # 32
]

# MediaPipe → YOLO 共享点索引映射：{mp_index: yolo_name}
MEDIAPIPE_SHARED_MAP = {
    0:  "nose",
    2:  "L_eye",
    5:  "R_eye",
    7:  "L_ear",
    8:  "R_ear",
    11: "L_sho",
    12: "R_sho",
    13: "L_elb",
    14: "R_elb",
    15: "L_wri",
    16: "R_wri",
    23: "L_hip",
    24: "R_hip",
    25: "L_kne",
    26: "R_kne",
    27: "L_ank",
    28: "R_ank",
}

# MediaPipe 独有点索引（不在 YOLO 中的）
MEDIAPIPE_UNIQUE_INDICES = [
    i for i in range(33) if i not in MEDIAPIPE_SHARED_MAP
]

# CSV 列命名：共享点用 YOLO 同名，独有点加 _mp 后缀
# 顺序：先放 17 个 YOLO 共享点（按 YOLO 顺序），再放 16 个 MediaPipe 独有点（按 MP 顺序）
def _build_mediapipe_csv_kpt_names() -> list[str]:
    """返回 33 个关键点的 CSV 列命名（与 MediaPipe 原始索引一一对应）。"""
    names = []
    for i in range(33):
        if i in MEDIAPIPE_SHARED_MAP:
            names.append(MEDIAPIPE_SHARED_MAP[i])   # 与 YOLO 同名
        else:
            names.append(f"{MEDIAPIPE_RAW_NAMES[i]}_mp")  # 加 _mp 后缀
    return names

MEDIAPIPE_CSV_KPT_NAMES = _build_mediapipe_csv_kpt_names()

# 推理时只读取 YOLO 兼容的 17 列（索引列表，相对于 33 点 CSV 列）
# 即 MEDIAPIPE_SHARED_MAP 的 key，按 YOLO 顺序排列
YOLO_COMPATIBLE_MP_INDICES = [
    mp_idx for mp_idx, _ in sorted(MEDIAPIPE_SHARED_MAP.items(), key=lambda kv: YOLO_KPT_NAMES.index(kv[1]))
]

# ── MediaPipe 骨架连接定义（33 点）──────────────────────────────────────
MEDIAPIPE_SKELETON_BONES = [
    # 脸部
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # 躯干
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    # 手部
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    # 下肢
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
    # 足部
    (29, 31), (30, 32),
]
