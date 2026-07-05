---
name: vlm-word-export
description: VLM分析报告导出Word：关键帧拼图(OpenCV) + python-docx生成 + VLM prompt引导"图N"引用
source: auto-skill
extracted_at: '2026-07-04T14:30:00.000Z'
---

# VLM 分析报告导出 Word

将 VLM 分析结果导出为 Word (.docx) 报告，包含关键帧拼图、编号引用和分析文本。

## 文件位置

- `utils/export_report.py` — 导出逻辑
- `core/vlm/analyzer.py` — VLM prompt 修改（引导"图N"引用）
- `gui/analysis_page.py` — 导出按钮集成

## 核心流程

```
VLM 分析完成 → sections[] 含 clip_b64 + figure_number
       │
       ▼
用户点击"导出报告" → QFileDialog.getSaveFileName()
       │
       ▼
export_vlm_report() → python-docx 生成 .docx
       │
       ├── 每个 section: clip_b64 → 抽帧 → 拼图 → 插入 Word
       └── 分析文本: Markdown 简单解析 → Word 段落
```

## 关键帧拼图

从 base64 视频中均匀抽帧，拼成网格图：

```python
def _extract_keyframes_from_b64(clip_b64: str, max_frames: int = 8) -> list[np.ndarray]:
    """base64 → 临时 MP4 → cv2.VideoCapture → 均匀抽帧"""
    data = base64.b64decode(clip_b64)
    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    try:
        tmp.write(data)
        tmp.close()
        cap = cv2.VideoCapture(tmp.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // max_frames)
        # 按 step 间隔抽帧...
    finally:
        os.unlink(tmp.name)

def _create_mosaic(frames: list[np.ndarray], cols: int = 4, target_width: int = 1200) -> np.ndarray:
    """多帧拼成 cols 列网格图"""
    rows = (len(frames) + cols - 1) // cols
    cell_w = target_width // cols
    cell_h = int(h * cell_w / w)
    canvas = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        resized = cv2.resize(frame, (cell_w, cell_h))
        canvas[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w] = resized
    return canvas
```

## Word 文档生成

```python
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

def export_vlm_report(output_path, sections, summary, infer_result=None, csv_name=""):
    doc = Document()
    
    # 标题
    title = doc.add_heading('跳绳动作分析报告', level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # 推理概要（可选）
    if infer_result:
        doc.add_heading('推理概要', level=1)
        # ... 判定结果、置信度、概率等
    
    # 逐问题分析
    for sec in sections:
        fig_num = sec['figure_number']
        doc.add_heading(sec['title'], level=2)
        
        # 元信息（灰色小字）
        p = doc.add_paragraph()
        run = p.add_run(f"帧范围：{start} - {end}")
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(128, 128, 128)
        
        # 关键帧拼图
        if clip_b64:
            frames = _extract_keyframes_from_b64(clip_b64)
            mosaic = _create_mosaic(frames, cols=4)
            # 图标题
            fig_caption = doc.add_paragraph()
            fig_caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = fig_caption.add_run(f"图{fig_num} 问题片段关键帧截图")
            run.font.size = Pt(10)
            # 插入图片
            tmp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            cv2.imwrite(tmp_img.name, mosaic)
            doc.add_picture(tmp_img.name, width=Inches(5.5))
            os.unlink(tmp_img.name)
        
        # 分析文本（Markdown 简单解析）
        _add_markdown_text(doc, sec['analysis'])
    
    doc.save(output_path)
```

## VLM Prompt 引导"图N"引用

修改 `SYSTEM_PROMPT` 和 `_build_image_contents()` 让 VLM 在分析文本中引用图编号：

```python
# SYSTEM_PROMPT 中添加：
# "在分析文本中，请用'如图N所示'来引用视频截图画面，其中N是给你的图编号。"

# _build_image_contents 中传入 figure_number：
def _build_image_contents(encoded_frames, prob_abnormal, figure_number):
    content = []
    for b64 in encoded_frames:
        content.append({'image': f'data:image/jpeg;base64,{b64}'})
    content.append({
        'text': f"这是图{figure_number}，...请在分析中用'如图{figure_number}所示'引用截图。"
    })
    return content
```

## Markdown → Word 文本转换

简单解析 Markdown 加粗/列表/标题语法：

```python
def _add_rich_text(paragraph, text: str):
    """解析 **bold** 语法"""
    parts = re.split(r'(\*\*.*?\*\*)', text)
    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            run = paragraph.add_run(part[2:-2])
            run.bold = True
        else:
            paragraph.add_run(part)

def _add_markdown_text(doc, text: str):
    for line in text.split('\n'):
        stripped = line.strip()
        if stripped.startswith('### '):
            doc.add_heading(stripped[4:], level=4)
        elif stripped.startswith('- '):
            p = doc.add_paragraph(style='List Bullet')
            _add_rich_text(p, stripped.lstrip('- '))
        else:
            p = doc.add_paragraph()
            _add_rich_text(p, stripped)
```

## 依赖

```
python-docx
```

## 踩坑点

1. **python-docx 中文字体**：需显式设置 `rPr.rFonts.set(eastAsia, '微软雅黑')`，否则中文可能显示为宋体或乱码。
2. **临时图片清理**：`doc.add_picture()` 后 Word 已读取文件内容，可立即 `os.unlink` 临时 PNG。
3. **拼图尺寸**：`target_width=1200` + `Inches(5.5)` 在 A4 纸上效果最佳，太宽会溢出页边距。
4. **VLM 不一定严格遵守"图N"引用**：prompt 中明确告知编号并在用户消息中重复强调可提高遵从率。
