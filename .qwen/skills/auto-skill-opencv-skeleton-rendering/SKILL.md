---
name: opencv-skeleton-rendering
description: OpenCV人体骨架渲染增强+YOLO姿态推理坑点：高可见度标记、分组配色、角度标注、千手观音修复、imgsz对齐、GPU加速
source: auto-skill
extracted_at: '2026-07-03T04:49:12.311Z'
---

# OpenCV 人体骨架渲染增强

在 1080×1920 或类似高分辨率画布上绘制人体骨架时，使用以下技巧提升可视化效果。

## 关键点标记（高可见度）

采用双层圆形：白底描边 + 彩色填充，对比度远高于单色圆点。

```python
# 参数（基于 1080×1920 画布）
KPT_OUTER_RADIUS = 14  # 白色外圈
KPT_INNER_RADIUS = 10  # 彩色内圈

# 绘制顺序：先白底，再彩色
cv2.circle(canvas, pt, KPT_OUTER_RADIUS, (255, 255, 255), -1, cv2.LINE_AA)
cv2.circle(canvas, pt, KPT_INNER_RADIUS, color, -1, cv2.LINE_AA)
```

## 骨骼连线（带阴影立体感）

每条骨骼线绘制两层：黑色阴影偏移 2px + 主线，增加深度感。

```python
BONE_THICKNESS = 4

# 阴影线（偏移 +2px）
shadow_pt1 = (pt1[0] + 2, pt1[1] + 2)
shadow_pt2 = (pt2[0] + 2, pt2[1] + 2)
cv2.line(canvas, shadow_pt1, shadow_pt2, (30, 30, 30), BONE_THICKNESS + 1, cv2.LINE_AA)

# 主线
cv2.line(canvas, pt1, pt2, bone_color, BONE_THICKNESS, cv2.LINE_AA)
```

## 肢体分组配色

按语义分组着色，一眼区分左右肢体和部位：

- **头部**：黄色系 `(0, 255, 255)`
- **左上肢**：绿色系 `(0, 255, 0)` ~ `(0, 220, 100)`
- **右上肢**：紫色系 `(255, 0, 255)` ~ `(200, 0, 255)`
- **左下肢**：橙色系 `(0, 140, 255)` ~ `(0, 80, 255)`
- **右下肢**：蓝色系 `(255, 140, 0)` ~ `(255, 80, 0)`
- **躯干**：白/灰 `(255, 255, 255)` / `(200, 200, 200)`

## 关节角度标注

在肘、膝、肩、髋等关节处绘制角度弧线及数值。

### 角度计算

```python
def calc_angle_deg(a, b, c):
    """计算向量 ba 与 bc 在顶点 b 处的夹角（度）"""
    ba = np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64)
    bc = np.array(c, dtype=np.float64) - np.array(b, dtype=np.float64)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
```

### 角度弧线绘制

使用 `cv2.ellipse` 绘制扇形填充 + 弧线边框，文字沿角平分线方向放置。

```python
ARC_RADIUS = 44

# 计算两向量相对于顶点的角度（OpenCV 坐标系：y 轴向下）
ang_a = math.degrees(math.atan2(pt_a[1] - vy, pt_a[0] - vx))
ang_c = math.degrees(math.atan2(pt_c[1] - vy, pt_c[0] - vx))

# 确保沿较短弧绘制
start_ang = min(ang_a, ang_c)
end_ang = max(ang_a, ang_c)
if end_ang - start_ang > 180:
    start_ang, end_ang = end_ang, start_ang + 360

# 半透明填充扇形（overlay 混合）
overlay = canvas.copy()
cv2.ellipse(overlay, vertex, (ARC_RADIUS, ARC_RADIUS),
            0, start_ang, end_ang, color, -1, cv2.LINE_AA)
cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

# 弧线边框
cv2.ellipse(canvas, vertex, (ARC_RADIUS, ARC_RADIUS),
            0, start_ang, end_ang, color, 2, cv2.LINE_AA)
```

### 文字放置（角平分线方向）

> **注意**：`cv2.FONT_HERSHEY_SIMPLEX` 仅支持 ASCII 字符，`°`（U+00B0）等非 ASCII 字符会渲染为 `??`。用 `deg` 替代。

```python
mid_ang = math.radians((start_ang + end_ang) / 2)
text_dist = ARC_RADIUS + 22
tx = int(vx + text_dist * math.cos(mid_ang))
ty = int(vy + text_dist * math.sin(mid_ang))

label = f"{angle_deg:.0f}deg"
(tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

# 黑底背景框
cv2.rectangle(canvas,
              (tx - 4, ty - th - 4),
              (tx + tw + 4, ty + baseline + 4),
              (0, 0, 0), -1)

# 文字
cv2.putText(canvas, label, (tx, ty), font, font_scale, color,
            font_thickness, cv2.LINE_AA)
```

## 常用角度关节定义（COCO 17 关键点）

```python
ANGLE_JOINTS = [
    ("L_Elb", 5,  7,  9,  color),   # 左肘：L_sho → L_elb → L_wri
    ("R_Elb", 6,  8,  10, color),   # 右肘
    ("L_Kne", 11, 13, 15, color),   # 左膝：L_hip → L_kne → L_ank
    ("R_Kne", 12, 14, 16, color),   # 右膝
    ("L_Shoulder", 7,  5,  11, color),  # 左肩
    ("R_Shoulder", 8,  6,  12, color),  # 右肩
    ("L_Hip", 5,  11, 13, color),   # 左髋
    ("R_Hip", 6,  12, 14, color),   # 右髋
]
```

---

## YOLO-Pose 推理常见坑点

### 千手观音（骨架重叠/鬼影）

**现象**：单人场景下骨架多层重叠，像"千手观音"。

**原因**：YOLO 对同一人检出多个 person 实例（NMS 未完全消除重叠框），全部绘制导致重叠。

**修复**：按平均关键点置信度排序，只保留得分最高的前 N 人：

```python
xy = result.keypoints.xy.cpu().numpy()       # [N, 17, 2]
conf = result.keypoints.conf.cpu().numpy()   # [N, 17]

num_persons = xy.shape[0]
mean_confs = np.array([conf[pid].mean() for pid in range(num_persons)])
ranked_ids = np.argsort(-mean_confs)[:max_persons]  # max_persons=1 用于单人场景
```

### imgsz 必须是 32 的倍数

**现象**：YOLO 报 `WARNING imgsz=[1080] must be multiple of max stride 32, updating to [1088]`

**修复**：传入 imgsz 前先向上取整至 32 的倍数：

```python
imgsz = ((target_width + 31) // 32) * 32
results = model(frame, verbose=False, imgsz=imgsz, device=device)
```

### GPU 加速推理

使用 `torch.cuda.is_available()` 自动选择设备，推理时显式传入 `device`：

```python
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = YOLO(model_path)
model.to(device)

results = model(frame, verbose=False, imgsz=imgsz, device=device)
```

推荐使用 `yolo11m-pose.pt` 或更大模型以获取更高精度（n-pose 在复杂场景下更易产生鬼影）。
