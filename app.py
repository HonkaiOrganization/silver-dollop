import os
import gradio as gr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core.extractor import PoseExtractor
from core.visualizer import visualize_pose
from core.infer import JumpRopeInference
from core.vlm import analyze_windows
import uuid

extractor = PoseExtractor()
inference = JumpRopeInference()

with gr.Blocks() as app:
    gr.Markdown("# WebUI Example")

    with gr.Tab("Video2CSV"):
        video_input = gr.Video(label="Input Video")
        convert_button = gr.Button("Convert Video")
        csv_output = gr.File(label="Output CSV", interactive=False)
        video_preview_csv = gr.Video(label="Video Preview", interactive=False)

    with gr.Tab("CsvVisualization"):
        csv_input = gr.File(label="Input CSV")
        visualize_button = gr.Button("Visualize CSV")
        video_preview = gr.Video(label="CSV Video Preview", interactive=False)

    with gr.Tab("JumpRopeInference"):
        infer_csv_input = gr.File(label="Input CSV(s) for Inference", file_count="multiple")
        infer_button = gr.Button("Run Inference")
        infer_status = gr.Textbox(label="Processing Status", interactive=False)
        infer_stats = gr.DataFrame(label="Detailed Statistics", interactive=False)
        infer_plot = gr.Plot(label="Abnormal Confidence Over Time")
        infer_json_output = gr.File(label="Inference Result (JSON)", interactive=False)

    with gr.Tab("VLM Analysis"):
        vlm_video = gr.Video(label="原始视频")
        vlm_json = gr.File(label="推理结果 (JSON)")
        vlm_button = gr.Button("开始 VLM 分析")
        vlm_status = gr.Textbox(label="分析状态", interactive=False)
        vlm_report = gr.Markdown(label="分析报告")

    def process_video(video, progress=gr.Progress()):
        if video is None:
            raise gr.Error("请先上传视频文件")
        output_csv = f"output/pose_data_{uuid.uuid4()}.csv"
        for i in extractor.extract_pose(video, output_csv):
            progress(i/2)
        output_video = f"output/skeleton_video_{uuid.uuid4()}.mp4"
        for i in visualize_pose(output_csv, output_video):
            progress(0.5 +i/2)
        yield output_csv,output_video

    def process_csv(csv, progress=gr.Progress()):
        if csv is None:
            raise gr.Error("请先上传CSV文件")
        output_video = f"output/skeleton_video_{uuid.uuid4()}.mp4"
        for i in visualize_pose(csv, output_video):
            progress(i)
        yield output_video

    convert_button.click(
        process_video,
        inputs=video_input,
            outputs=[csv_output, video_preview_csv],
        )
    visualize_button.click(
        process_csv,
        inputs=csv_input,
        outputs=[video_preview],
    )
    def run_inference(csv_files):
        if not csv_files:
            raise gr.Error("请先上传CSV文件")

        # Gradio file_count="multiple" returns a list of file paths
        if isinstance(csv_files, str):
            csv_files = [csv_files]

        output_json = f"output/inference_result_{uuid.uuid4()}.json"
        all_results = {}
        status_lines = []

        for i, csv_path in enumerate(csv_files, 1):
            fname = os.path.basename(csv_path)
            status_lines.append(f"[{i}/{len(csv_files)}] 正在处理: {fname}")
            try:
                result = inference.predict(csv_path, output_json_path=output_json)
                for r_fname, r_data in result["results"].items():
                    all_results[r_fname] = r_data
                r_ok = result["results"].get(fname)
                if r_ok and r_ok.get("status") == "ok":
                    status_lines.append(f"  -> {r_ok['predicted_label']} (confidence: {r_ok['confidence']:.4f})")
                else:
                    err = r_ok.get("reason", r_ok.get("error", "unknown")) if r_ok else "no result"
                    status_lines.append(f"  -> 失败: {err}")
            except Exception as e:
                status_lines.append(f"  -> 错误: {str(e)}")
                all_results[fname] = {"status": "error", "error": str(e)}

        rows = []
        for fname, r in all_results.items():
            if r.get("status") == "ok":
                rows.append({
                    "file": fname,
                    "label": r["predicted_label"],
                    "confidence": f'{r["confidence"]:.4f}',
                    "P(normal)": f'{r["probabilities"]["normal"]:.4f}',
                    "P(abnormal)": f'{r["probabilities"]["abnormal"]:.4f}',
                    "windows": r["num_windows"],
                    "frames": r["num_frames"],
                })
            else:
                rows.append({
                    "file": fname,
                    "label": r.get("status", "unknown"),
                    "confidence": "-",
                    "P(normal)": "-",
                    "P(abnormal)": "-",
                    "windows": "-",
                    "frames": r.get("reason", r.get("error", "-")),
                })

        stats_df = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(10, 4))
        has_details = False
        for fname, r in all_results.items():
            if r.get("status") == "ok" and "window_details" in r:
                details = r["window_details"]
                x = [d["start_frame"] for d in details]
                y = [d["prob_abnormal"] for d in details]
                ax.plot(x, y, marker=".", label=fname)
                has_details = True
        if has_details:
            ax.set_xlabel("Frame")
            ax.set_ylabel("P(abnormal)")
            ax.set_title("Abnormal Confidence Per Sliding Window")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No window details available", ha="center", va="center", transform=ax.transAxes)

        fig.tight_layout()

        return "\n".join(status_lines), stats_df, fig, output_json

    infer_button.click(
        run_inference,
        inputs=infer_csv_input,
        outputs=[infer_status, infer_stats, infer_plot, infer_json_output],
    )

    def run_vlm_analysis(video, json_file):
        if video is None:
            raise gr.Error("请先上传原始视频文件")
        if json_file is None:
            raise gr.Error("请先上传推理结果 JSON 文件")

        result = analyze_windows(video, json_file, top_k=3)
        return "分析完成", result["markdown"]

    vlm_button.click(
        run_vlm_analysis,
        inputs=[vlm_video, vlm_json],
        outputs=[vlm_status, vlm_report],
    )

if __name__ == "__main__":
    app.launch()