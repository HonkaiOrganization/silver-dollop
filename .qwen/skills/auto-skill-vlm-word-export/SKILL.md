---
name: vlm-word-export
description: VLM analysis report export to Word: keyframe mosaic (OpenCV) + python-docx generation + VLM prompt-guided "Figure N" references
source: auto-skill
extracted_at: '2026-07-04T14:30:00.000Z'
---

# VLM Analysis Report Export to Word

Export VLM analysis results into a Word (.docx) report, including keyframe mosaics, numbered figure references, and analysis text.

## File Locations

- `utils/export_report.py` — Export logic
- `core/vlm/analyzer.py` — VLM prompt modifications (guide "Figure N" references)
- `gui/analysis_page.py` — Export button integration

## Core Workflow

```
VLM analysis complete → sections[] contains clip_b64 + figure_number
       │
       ▼
User clicks "Export Report" → QFileDialog.getSaveFileName()
       │
       ▼
export_vlm_report() → python-docx generates .docx
       │
       ├── Each section: clip_b64 → extract frames → mosaic → insert into Word
       └── Analysis text: Markdown simple parsing → Word paragraphs
```

## Keyframe Mosaic

Uniformly sample frames from base64-encoded video and assemble them into a grid mosaic:

```python
def _extract_keyframes_from_b64(clip_b64: str, max_frames: int = 8) -> list[np.ndarray]:
    """base64 → temporary MP4 → cv2.VideoCapture → uniform frame sampling"""
    data = base64.b64decode(clip_b64)
    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    try:
        tmp.write(data)
        tmp.close()
        cap = cv2.VideoCapture(tmp.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // max_frames)
        # Sample frames at step intervals...
    finally:
        os.unlink(tmp.name)

def _create_mosaic(frames: list[np.ndarray], cols: int = 4, target_width: int = 1200) -> np.ndarray:
    """Assemble multiple frames into a grid mosaic with cols columns"""
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

## Word Document Generation

```python
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

def export_vlm_report(output_path, sections, summary, infer_result=None, csv_name=""):
    doc = Document()

    # Title
    title = doc.add_heading('Jump Rope Motion Analysis Report', level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Inference summary (optional)
    if infer_result:
        doc.add_heading('Inference Summary', level=1)
        # ... classification result, confidence, probabilities, etc.

    # Per-issue analysis
    for sec in sections:
        fig_num = sec['figure_number']
        doc.add_heading(sec['title'], level=2)

        # Metadata (gray small text)
        p = doc.add_paragraph()
        run = p.add_run(f"Frame range: {start} - {end}")
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(128, 128, 128)

        # Keyframe mosaic
        if clip_b64:
            frames = _extract_keyframes_from_b64(clip_b64)
            mosaic = _create_mosaic(frames, cols=4)
            # Figure caption
            fig_caption = doc.add_paragraph()
            fig_caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = fig_caption.add_run(f"Figure {fig_num} Keyframe screenshots of the issue segment")
            run.font.size = Pt(10)
            # Insert image
            tmp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            cv2.imwrite(tmp_img.name, mosaic)
            doc.add_picture(tmp_img.name, width=Inches(5.5))
            os.unlink(tmp_img.name)

        # Analysis text (Markdown simple parsing)
        _add_markdown_text(doc, sec['analysis'])

    doc.save(output_path)
```

## VLM Prompt Guiding "Figure N" References

Modify `SYSTEM_PROMPT` and `_build_image_contents()` so the VLM references figure numbers in its analysis text:

```python
# Add to SYSTEM_PROMPT:
# "In the analysis text, use 'as shown in Figure N' to reference the video screenshots, where N is the figure number assigned to you."

# Pass figure_number in _build_image_contents:
def _build_image_contents(encoded_frames, prob_abnormal, figure_number):
    content = []
    for b64 in encoded_frames:
        content.append({'image': f'data:image/jpeg;base64,{b64}'})
    content.append({
        'text': f"This is Figure {figure_number}, ...please reference the screenshots using 'as shown in Figure {figure_number}' in your analysis."
    })
    return content
```

## Markdown to Word Text Conversion

Simple parsing of Markdown bold, list, and heading syntax:

```python
def _add_rich_text(paragraph, text: str):
    """Parse **bold** syntax"""
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

## Dependencies

```
python-docx
```

## Pitfalls and Gotchas

1. **python-docx CJK font**: You must explicitly set `rPr.rFonts.set(eastAsia, 'Microsoft YaHei')`; otherwise CJK characters may render in SimSun or appear as garbled text.
2. **Temporary image cleanup**: After `doc.add_picture()`, Word has already read the file contents into the document, so the temporary PNG can be immediately deleted with `os.unlink`.
3. **Mosaic dimensions**: `target_width=1200` combined with `Inches(5.5)` produces the best result on A4 paper; wider values will overflow the page margins.
4. **VLM may not strictly follow "Figure N" references**: Explicitly stating the figure number in the prompt and reiterating it in the user message improves compliance.
