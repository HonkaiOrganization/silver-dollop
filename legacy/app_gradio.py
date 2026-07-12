import os
import gradio as gr
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.pose import PoseProcessor
from config import KPT_NAMES, CSV_COLUMNS
from core.infer import JumpRopeInference
from core.vlm import analyze_windows
import uuid

pose_processor = PoseProcessor()
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
        vlm_video = gr.Video(label="Source Video")
        vlm_json = gr.File(label="Inference Result (JSON)")
        vlm_button = gr.Button("Start VLM Analysis")
        vlm_status = gr.Textbox(label="Analysis Status", interactive=False)
        vlm_report = gr.Markdown(label="Analysis Report")

    def _extract_pose_to_csv(video_path: str, csv_path: str, target_size=(1080, 1920)):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        all_data = []
        fid = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
            result = pose_processor.process(frame, target_size=target_size)
            kpts = result.get("keypoints_array", {})
            xy, conf = kpts.get("xy"), kpts.get("conf")
            if xy is not None and conf is not None:
                row = [float(fid), 0.0]
                for i in range(17):
                    row.extend([float(xy[i, 0]), float(xy[i, 1]), float(conf[i])])
                all_data.append(row)
            fid += 1
            yield fid / max(total, 1)
        cap.release()
        df = pd.DataFrame(all_data, columns=CSV_COLUMNS) if all_data else pd.DataFrame(columns=CSV_COLUMNS)
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        df.to_csv(csv_path, index=False)

    def _render_skeleton_video(csv_path: str, video_path: str, width=1080, height=1920, fps=30.0):
        df = pd.read_csv(csv_path)
        x_cols = [f"{n}_x" for n in KPT_NAMES]
        y_cols = [f"{n}_y" for n in KPT_NAMES]
        c_cols = [f"{n}_conf" for n in KPT_NAMES]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        frame_ids = sorted(df['frame_id'].unique())
        total = len(frame_ids)
        for idx, fid in enumerate(frame_ids):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            rows = df[df['frame_id'] == fid]
            if not rows.empty:
                row = rows.iloc[0]
                xs = np.nan_to_num(row[x_cols].values, nan=0.0)
                ys = np.nan_to_num(row[y_cols].values, nan=0.0)
                confs = np.nan_to_num(row[c_cols].values, nan=0.0)
                xy = np.stack([xs, ys], axis=1)
                PoseProcessor.render_skeleton(canvas, xy, confs)
            writer.write(canvas)
            yield (idx + 1) / max(total, 1)
        writer.release()

    def process_video(video, progress=gr.Progress()):
        if video is None:
            raise gr.Error("Please upload a video file first")
        output_csv = f"output/pose_data_{uuid.uuid4()}.csv"
        for p in _extract_pose_to_csv(video, output_csv):
            progress(p / 2)
        output_video = f"output/skeleton_video_{uuid.uuid4()}.mp4"
        for p in _render_skeleton_video(output_csv, output_video):
            progress(0.5 + p / 2)
        yield output_csv, output_video

    def process_csv(csv, progress=gr.Progress()):
        if csv is None:
            raise gr.Error("Please upload a CSV file first")
        output_video = f"output/skeleton_video_{uuid.uuid4()}.mp4"
        for p in _render_skeleton_video(csv, output_video):
            progress(p)
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
            raise gr.Error("Please upload a CSV file first")

        # Gradio file_count="multiple" returns a list of file paths
        if isinstance(csv_files, str):
            csv_files = [csv_files]

        output_json = f"output/inference_result_{uuid.uuid4()}.json"
        all_results = {}
        status_lines = []

        for i, csv_path in enumerate(csv_files, 1):
            fname = os.path.basename(csv_path)
            status_lines.append(f"[{i}/{len(csv_files)}] Processing: {fname}")
            try:
                result = inference.predict(csv_path, output_json_path=output_json)
                for r_fname, r_data in result["results"].items():
                    all_results[r_fname] = r_data
                r_ok = result["results"].get(fname)
                if r_ok and r_ok.get("status") == "ok":
                    status_lines.append(f"  -> {r_ok['predicted_label']} (confidence: {r_ok['confidence']:.4f})")
                else:
                    err = r_ok.get("reason", r_ok.get("error", "unknown")) if r_ok else "no result"
                    status_lines.append(f"  -> Failed: {err}")
            except Exception as e:
                status_lines.append(f"  -> Error: {str(e)}")
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
            raise gr.Error("Please upload the source video file first")
        if json_file is None:
            raise gr.Error("Please upload the inference result JSON file first")

        result = analyze_windows(video, json_file, top_k=3)
        return "Analysis complete", result["markdown"]

    vlm_button.click(
        run_vlm_analysis,
        inputs=[vlm_video, vlm_json],
        outputs=[vlm_status, vlm_report],
    )

if __name__ == "__main__":
    app.launch()