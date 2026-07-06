---
name: opencv-skeleton-rendering
description: OpenCV human skeleton rendering enhancements + YOLO pose inference pitfalls: high-visibility markers, grouped coloring, angle annotation, multi-instance ghosting fix, imgsz alignment, GPU acceleration
source: auto-skill
extracted_at: '2026-07-03T04:49:12.311Z'
---

# OpenCV Human Skeleton Rendering Enhancements

When drawing human skeletons on a 1080x1920 or similar high-resolution canvas, use the following techniques to improve visual quality.

## Keypoint Markers (High Visibility)

Use a two-layer circle: white outer stroke + colored inner fill, which provides much higher contrast than a single-color dot.

```python
# Parameters (based on a 1080x1920 canvas)
KPT_OUTER_RADIUS = 14  # White outer ring
KPT_INNER_RADIUS = 10  # Colored inner ring

# Drawing order: white base first, then colored fill
cv2.circle(canvas, pt, KPT_OUTER_RADIUS, (255, 255, 255), -1, cv2.LINE_AA)
cv2.circle(canvas, pt, KPT_INNER_RADIUS, color, -1, cv2.LINE_AA)
```

## Bone Lines (with Shadow for Depth)

Draw each bone line in two layers: a black shadow offset by 2px + the main line, to add a sense of depth.

```python
BONE_THICKNESS = 4

# Shadow line (offset +2px)
shadow_pt1 = (pt1[0] + 2, pt1[1] + 2)
shadow_pt2 = (pt2[0] + 2, pt2[1] + 2)
cv2.line(canvas, shadow_pt1, shadow_pt2, (30, 30, 30), BONE_THICKNESS + 1, cv2.LINE_AA)

# Main line
cv2.line(canvas, pt1, pt2, bone_color, BONE_THICKNESS, cv2.LINE_AA)
```

## Limb Group Coloring

Color by semantic groups to visually distinguish left/right limbs and body regions at a glance:

- **Head**: Yellow `(0, 255, 255)`
- **Left upper limb**: Green `(0, 255, 0)` ~ `(0, 220, 100)`
- **Right upper limb**: Purple `(255, 0, 255)` ~ `(200, 0, 255)`
- **Left lower limb**: Orange `(0, 140, 255)` ~ `(0, 80, 255)`
- **Right lower limb**: Blue `(255, 140, 0)` ~ `(255, 80, 0)`
- **Torso**: White/Gray `(255, 255, 255)` / `(200, 200, 200)`

## Joint Angle Annotation

Draw angle arcs and numeric values at joints such as elbows, knees, shoulders, and hips.

### Angle Calculation

```python
def calc_angle_deg(a, b, c):
    """Calculate the angle (in degrees) between vectors ba and bc at vertex b."""
    ba = np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64)
    bc = np.array(c, dtype=np.float64) - np.array(b, dtype=np.float64)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
```

### Angle Arc Drawing

Use `cv2.ellipse` to draw a filled sector + arc border, with text placed along the angle bisector direction.

```python
ARC_RADIUS = 44

# Compute the angles of the two vectors relative to the vertex (OpenCV coordinate system: y-axis points downward)
ang_a = math.degrees(math.atan2(pt_a[1] - vy, pt_a[0] - vx))
ang_c = math.degrees(math.atan2(pt_c[1] - vy, pt_c[0] - vx))

# Ensure drawing along the shorter arc
start_ang = min(ang_a, ang_c)
end_ang = max(ang_a, ang_c)
if end_ang - start_ang > 180:
    start_ang, end_ang = end_ang, start_ang + 360

# Semi-transparent filled sector (overlay blending)
overlay = canvas.copy()
cv2.ellipse(overlay, vertex, (ARC_RADIUS, ARC_RADIUS),
            0, start_ang, end_ang, color, -1, cv2.LINE_AA)
cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

# Arc border
cv2.ellipse(canvas, vertex, (ARC_RADIUS, ARC_RADIUS),
            0, start_ang, end_ang, color, 2, cv2.LINE_AA)
```

### Text Placement (along the angle bisector)

> **Note**: `cv2.FONT_HERSHEY_SIMPLEX` only supports ASCII characters. Non-ASCII characters such as `deg` (U+00B0) will render as `??`. Use `deg` as a substitute.

```python
mid_ang = math.radians((start_ang + end_ang) / 2)
text_dist = ARC_RADIUS + 22
tx = int(vx + text_dist * math.cos(mid_ang))
ty = int(vy + text_dist * math.sin(mid_ang))

label = f"{angle_deg:.0f}deg"
(tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

# Black background box
cv2.rectangle(canvas,
              (tx - 4, ty - th - 4),
              (tx + tw + 4, ty + baseline + 4),
              (0, 0, 0), -1)

# Text
cv2.putText(canvas, label, (tx, ty), font, font_scale, color,
            font_thickness, cv2.LINE_AA)
```

## Common Angle Joint Definitions (COCO 17 Keypoints)

```python
ANGLE_JOINTS = [
    ("L_Elb", 5,  7,  9,  color),   # Left elbow: L_sho -> L_elb -> L_wri
    ("R_Elb", 6,  8,  10, color),   # Right elbow
    ("L_Kne", 11, 13, 15, color),   # Left knee: L_hip -> L_kne -> L_ank
    ("R_Kne", 12, 14, 16, color),   # Right knee
    ("L_Shoulder", 7,  5,  11, color),  # Left shoulder
    ("R_Shoulder", 8,  6,  12, color),  # Right shoulder
    ("L_Hip", 5,  11, 13, color),   # Left hip
    ("R_Hip", 6,  12, 14, color),   # Right hip
]
```

---

## Common YOLO-Pose Inference Pitfalls

### Multi-Instance Ghosting (Skeleton Overlap)

**Symptom**: In a single-person scene, skeletons overlap in multiple layers, producing a "thousand-armed" effect.

**Cause**: YOLO detects multiple person instances for the same individual (NMS does not fully eliminate overlapping boxes), and drawing all of them causes visual overlap.

**Fix**: Sort by mean keypoint confidence and keep only the top N highest-scoring detections:

```python
xy = result.keypoints.xy.cpu().numpy()       # [N, 17, 2]
conf = result.keypoints.conf.cpu().numpy()   # [N, 17]

num_persons = xy.shape[0]
mean_confs = np.array([conf[pid].mean() for pid in range(num_persons)])
ranked_ids = np.argsort(-mean_confs)[:max_persons]  # max_persons=1 for single-person scenes
```

### imgsz Must Be a Multiple of 32

**Symptom**: YOLO reports `WARNING imgsz=[1080] must be multiple of max stride 32, updating to [1088]`

**Fix**: Round up to the nearest multiple of 32 before passing imgsz:

```python
imgsz = ((target_width + 31) // 32) * 32
results = model(frame, verbose=False, imgsz=imgsz, device=device)
```

### GPU-Accelerated Inference

Use `torch.cuda.is_available()` to automatically select the device, and explicitly pass `device` during inference:

```python
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YOLO(model_path)
model.to(device)

results = model(frame, verbose=False, imgsz=imgsz, device=device)
```

It is recommended to use `yolo11m-pose.pt` or a larger model for higher accuracy (n-pose is more prone to ghosting in complex scenes).
