"""
Global configuration module.
Pose engine selection via POSE_ENGINE environment variable.
Supported values:
  - "YOLO"       (default) YOLO-Pose
  - "MediaPipe"  Google MediaPipe Pose
"""
import os
import logging

# -- Logging configuration ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

# -- Common constants: YOLO COCO 17 keypoints --------------------------------
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

# -- Pose engine -------------------------------------------------------------
POSE_ENGINE: str = os.getenv("POSE_ENGINE", "YOLO").strip()
assert POSE_ENGINE in ("YOLO", "MediaPipe"), (
    f"Invalid POSE_ENGINE value: {POSE_ENGINE!r}, only 'YOLO' or 'MediaPipe' are supported"
)

# -- MediaPipe Tasks API model path -------------------------------------------
# Tasks API requires .task model files, auto-downloaded on first use.
# Available models:
#   pose_landmarker_lite  -- fastest, lower accuracy (~5 MB)
#   pose_landmarker_full  -- balanced, higher accuracy (~30 MB)  <-- default
#   pose_landmarker_heavy -- slowest, highest accuracy (~50 MB)
MEDIAPIPE_MODEL_PATH: str = os.getenv(
    "MEDIAPIPE_MODEL_PATH",
    "pretrained/pose_landmarker_full.task",
).strip()

MEDIAPIPE_MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)

# -- MediaPipe-to-YOLO keypoint mapping --------------------------------------
# YOLO COCO 17 keypoint names (kept in sync with KPT_NAMES)
YOLO_KPT_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_sho", "R_sho", "L_elb", "R_elb",
    "L_wri", "R_wri", "L_hip", "R_hip",
    "L_kne", "R_kne", "L_ank", "R_ank",
]

# MediaPipe 33 raw keypoint names
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

# MediaPipe -> YOLO shared keypoint index mapping: {mp_index: yolo_name}
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

# MediaPipe unique keypoint indices (not present in YOLO)
MEDIAPIPE_UNIQUE_INDICES = [
    i for i in range(33) if i not in MEDIAPIPE_SHARED_MAP
]

# CSV column naming: shared keypoints use YOLO names, unique ones get _mp suffix.
# Order: 17 YOLO-shared keypoints (in YOLO order), then 16 MediaPipe-unique (in MP order).
def _build_mediapipe_csv_kpt_names() -> list[str]:
    """Return 33 keypoint CSV column names (1:1 mapping with MediaPipe raw indices)."""
    names = []
    for i in range(33):
        if i in MEDIAPIPE_SHARED_MAP:
            names.append(MEDIAPIPE_SHARED_MAP[i])   # same name as YOLO
        else:
            names.append(f"{MEDIAPIPE_RAW_NAMES[i]}_mp")  # append _mp suffix
    return names

MEDIAPIPE_CSV_KPT_NAMES = _build_mediapipe_csv_kpt_names()

# Inference reads only the YOLO-compatible 17 columns (index list relative to 33-point CSV).
# These are MEDIAPIPE_SHARED_MAP keys sorted by YOLO keypoint order.
YOLO_COMPATIBLE_MP_INDICES = [
    mp_idx for mp_idx, _ in sorted(MEDIAPIPE_SHARED_MAP.items(), key=lambda kv: YOLO_KPT_NAMES.index(kv[1]))
]

# -- MediaPipe skeleton connectivity (33 keypoints) --------------------------
MEDIAPIPE_SKELETON_BONES = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Torso
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    # Hands
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    # Lower limbs
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
    # Feet
    (29, 31), (30, 32),
]
