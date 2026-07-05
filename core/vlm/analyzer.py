import os
import json
import logging
import tempfile
import base64
import cv2
import requests
import dashscope
from dashscope import MultiModalConversation

dashscope.base_http_api_url = "https://llm-ory446glox99u6xk.cn-beijing.maas.aliyuncs.com/api/v1"

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一名专业的跳绳运动姿态分析专家。你将看到跳绳视频中被模型判定为异常的片段截图，以及该片段被判定为异常的概率。

请根据截图中跳绳者的真实动作画面，从运动生物力学和跳绳专项技术角度，分析该片段中存在的问题。

分析维度包括：
1. 身体姿态：躯干是否过度前倾/后仰，重心是否稳定，身体是否保持直立
2. 肢体协调：上下肢动作是否协调，摆臂与起跳节奏是否匹配
3. 关节角度：膝、髋、踝关节在起跳和落地时的角度是否合理，是否有过度屈伸
4. 跳绳技术：摇绳轨迹是否规范，起跳时机是否准确，落地是否有缓冲，双脚是否同步
5. 常见错误：是否存在双摇失误、绊绳、跳得过高或过低、手臂外展过大等问题

请严格按照以下格式输出，不要增减标题，不要改变结构：

**问题描述**
（在此描述观察到的异常姿态，引用截图时使用"如图N所示"，其中N是给你的图编号）

**可能原因**
（在此分析导致该问题的技术原因）

**改进建议**
（在此给出针对性的训练改进建议）"""


def _slice_window_frames(video_path: str, start_frame: int, end_frame: int, max_frames: int = 8) -> list[str]:
    """从视频中截取指定窗口范围的帧，均匀采样最多max_frames帧，返回base64编码列表。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, start_frame)
    end_frame = min(end_frame, total)

    if start_frame >= end_frame:
        cap.release()
        raise ValueError(f"窗口帧范围无效: {start_frame}-{end_frame}")

    indices = _linspace(start_frame, end_frame - 1, min(end_frame - start_frame, max_frames))
    encoded = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        encoded.append(base64.b64encode(buf.tobytes()).decode('utf-8'))

    cap.release()
    return encoded


def _linspace(start: int, stop: int, num: int) -> list[int]:
    if num <= 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [int(round(start + step * i)) for i in range(num)]


def _build_image_contents(encoded_frames: list[str], prob_abnormal: float, figure_number: int) -> list[dict]:
    """构建多模态消息的content列表：多张图 + 文本描述。"""
    content = []
    for i, b64 in enumerate(encoded_frames):
        content.append({'image': f'data:image/jpeg;base64,{b64}'})
    content.append({
        'text': (
            f"这是图{figure_number}，包含该片段共 {len(encoded_frames)} 帧关键帧截图，"
            f"模型判定为异常的概率为 {prob_abnormal:.4f}。\n"
            "请分析这些帧中跳绳者的姿态问题，并给出改进建议。"
            "在分析文本中请用\"如图{figure_number}所示\"来引用截图画面。"
        )
    })
    return content


def _call_vlm(messages: list[dict], api_key: str | None = None) -> str:
    response = MultiModalConversation.call(
        api_key=api_key,  # pyright: ignore[reportArgumentType]
        model='qwen3.7-plus',
        messages=messages,
    )
    if response.status_code == 200:  # type: ignore
        return response.output.choices[0].message.content[0]["text"]  # type: ignore
    last_error = f"status_code={response.status_code}"  # type: ignore
    return last_error


def _extract_clip_b64(
    video_path: str,
    start_frame: int,
    end_frame: int,
    resolution_h: int = 240,
    target_fps: int = 10,
) -> str | None:
    """从视频中截取指定帧范围的片段，编码为 base64 MP4。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = max(0, start_frame)
        end_frame = min(end_frame, total)
        if start_frame >= end_frame:
            return None

        frame_count = end_frame - start_frame
        if frame_count < 2:
            return None

        step = max(1, int(frame_count / (target_fps * 2.0)))
        indices = list(range(start_frame, end_frame, step))
        if not indices:
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, indices[0])
        ret, first_frame = cap.read()
        if not ret:
            return None

        h_orig, w_orig = first_frame.shape[:2]
        scale = resolution_h / h_orig
        w_out = int(w_orig * scale) // 2 * 2
        h_out = resolution_h // 2 * 2

        tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        tmp.close()

        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(tmp.name, fourcc, target_fps, (w_out, h_out))
            if not writer.isOpened():
                os.unlink(tmp.name)
                return None

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                resized = cv2.resize(frame, (w_out, h_out))
                writer.write(resized)
            writer.release()
        finally:
            cap.release()

        if not os.path.exists(tmp.name) or os.path.getsize(tmp.name) == 0:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
            return None

        with open(tmp.name, 'rb') as f:
            clip_b64 = base64.b64encode(f.read()).decode('ascii')
        os.unlink(tmp.name)
        return clip_b64
    except Exception:
        cap.release()
        return None


def analyze_windows(
    video_path: str,
    json_path: str,
    top_k: int = 3,
    api_key: str | None = None,
    max_frames: int = 8,
    clip_params: dict | None = None,
) -> dict:
    """
    根据推理 JSON 中 window_details 的 prob_abnormal 排名，
    截取 top-k 最有问题的窗口帧，调用 VLM 逐一分析。

    Returns:
        dict: {"markdown": str, "sections": [...], "summary": str}
    """
    if clip_params is None:
        clip_params = {}

    if api_key is None:
        api_key = os.getenv('DASHSCOPE_API_KEY', None)
        logger.info("使用环境变量 DASHSCOPE_API_KEY: %s", api_key)
        if api_key is None:
            return {
                "markdown": "VLM 分析失败：未提供 DASHSCOPE_API_KEY。",
                "sections": [],
                "summary": "",
            }

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_windows = []
    results = data.get('results', {})
    for fname, r in results.items():
        if r.get('status') == 'ok' and 'window_details' in r:
            for w in r['window_details']:
                w['_file'] = fname
                all_windows.append(w)

    if not all_windows:
        return {
            "markdown": "# VLM 分析报告\n\n未找到有效的窗口分析数据。",
            "sections": [],
            "summary": "",
        }

    all_windows.sort(key=lambda w: w['prob_abnormal'], reverse=True)
    top_windows = all_windows[:top_k]

    md_sections = [
        "# 跳绳动作 VLM 分析报告",
        "",
        f"共分析 **{len(top_windows)}** 个最异常窗口（按 `prob_abnormal` 降序选取）。",
        "",
    ]
    sections = []

    for idx, w in enumerate(top_windows, 1):
        fname = w.pop('_file')
        start = w['start_frame']
        end = w['end_frame']
        prob = w['prob_abnormal']
        title = f"问题片段 {idx} / {len(top_windows)}"

        md_sections.append(f"## {title}")
        md_sections.append(f"- **来源文件**: `{fname}`")
        md_sections.append(f"- **帧范围**: {start} - {end}")
        md_sections.append(f"- **异常概率**: {prob:.4f}")
        md_sections.append("")

        try:
            encoded = _slice_window_frames(video_path, start, end, max_frames=max_frames)
        except Exception as e:
            md_sections.append(f"> ⚠️ 截取帧失败: {e}")
            md_sections.append("")
            continue

        if not encoded:
            md_sections.append("> ⚠️ 未提取到有效帧")
            md_sections.append("")
            continue

        clip_b64 = _extract_clip_b64(
            video_path, start, end,
            resolution_h=clip_params.get('resolution_h', 240),
            target_fps=clip_params.get('target_fps', 10),
        )

        messages = [
            {'role': 'system', 'content': [{'text': SYSTEM_PROMPT}]},
            {'role': 'user', 'content': _build_image_contents(encoded, prob, idx)},
        ]

        analysis = _call_vlm(messages, api_key=api_key)
        md_sections.append(analysis)
        md_sections.append("")
        md_sections.append("---")
        md_sections.append("")

        sections.append({
            "title": title,
            "prob": prob,
            "start_frame": start,
            "end_frame": end,
            "analysis": analysis,
            "clip_b64": clip_b64,
            "figure_number": idx,
        })

    summary_md = (
        "以上按异常概率从高到低列出了最有问题的 {k} 个片段。"
        "建议优先关注 **片段 1** 中的姿态问题，依次改进。".format(k=len(top_windows))
    )
    md_sections.append("## 总结与改进优先级")
    md_sections.append("")
    md_sections.append(summary_md)
    md_sections.append("")

    return {
        "markdown": "\n".join(md_sections),
        "sections": sections,
        "summary": summary_md,
    }


if __name__ == "__main__":
    import click
    from pathlib import Path

    @click.command()
    @click.option('--video', required=True, type=click.Path(exists=True), help='跳绳视频文件路径')
    @click.option('--json', 'json_path', required=True, type=click.Path(exists=True), help='JumpRopeInference 输出的 JSON 文件路径')
    @click.option('--top_k', default=3, type=int, help='选取 prob_abnormal 最高的窗口数量 (默认: 3)')
    @click.option('--max_frames', default=8, type=int, help='每个片段截取的最多帧数 (默认: 8)')
    @click.option('--api-key', default=None, help='DashScope API Key，可覆盖环境变量 DASHSCOPE_API_KEY')
    @click.option('--output', default='vlm_report.md', type=click.Path(), help='分析结果输出文件路径')
    def main(video, json_path, top_k, max_frames, api_key, output):
        api_key = api_key or os.getenv('DASHSCOPE_API_KEY')
        result = analyze_windows(
            video_path=video,
            json_path=json_path,
            top_k=top_k,
            api_key=api_key,
            max_frames=max_frames,
        )

        output_path = Path(output)
        output_path.write_text(result["markdown"], encoding='utf-8')
        print(f'分析完成，报告已写入: {output_path}')

    main()
