import os
import json
import glob
import logging
import torch
import torch.nn.functional as F
import numpy as np

from utils.load_csv import load_and_normalize_csv
from models.model import JumpRopeClassifier

logger = logging.getLogger(__name__)

LABEL_MAP = {0: 'abnormal', 1: 'normal'}


class JumpRopeInference:
    def __init__(self, model_path: str = 'pretrained/model_export.pt'):
        """
        Initialize inference engine and load model.

        Args:
            model_path: Exported model file path (.pt)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        export_data = torch.load(model_path, map_location=self.device, weights_only=True)
        model_cfg = export_data['config']

        self.model = JumpRopeClassifier(
            in_channels=model_cfg['in_channels'],
            num_classes=model_cfg['num_classes']
        ).to(self.device)
        self.model.load_state_dict(export_data['model_state_dict'])
        self.model.eval()

        self.window_size = model_cfg['window_size']
        self.stride = model_cfg.get('stride', 32)

    def _segment_and_predict(self, coords: np.ndarray) -> dict:
        """
        Perform sliding window inference on a single video segment.
        """
        num_frames = coords.shape[0]
        if num_frames < self.window_size:
            return {
                'status': 'skip',
                'reason': f'Insufficient frames count: {num_frames} < {self.window_size}'
            }

        window_probs = []
        window_details = []
        for i in range(0, num_frames - self.window_size + 1, self.stride):
            window = coords[i:i + self.window_size]
            window_flat = window.reshape(self.window_size, -1)
            input_tensor = torch.tensor(window_flat, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = F.softmax(logits, dim=1)
                probs_np = probs.cpu().numpy()[0]
                window_probs.append(probs_np)
                window_details.append({
                    'window_index': len(window_details),
                    'start_frame': int(i),
                    'end_frame': int(i + self.window_size),
                    'prob_abnormal': float(probs_np[0]),
                    'prob_normal': float(probs_np[1]),
                })

        avg_probs = np.mean(window_probs, axis=0)
        predicted_class = int(np.argmax(avg_probs))
        confidence = float(avg_probs[predicted_class])

        return {
            'status': 'ok',
            'predicted_label': LABEL_MAP[predicted_class],
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': {
                'abnormal': float(avg_probs[0]),
                'normal': float(avg_probs[1])
            },
            'num_windows': len(window_probs),
            'num_frames': num_frames,
            'window_details': window_details,
        }

    def predict(self, input_path: str, output_json_path: str | None = None) -> dict:
        """
        Perform batch inference on CSV files or directories.

        Args:
            input_path: Input CSV file or directory path
            output_json_path: Output JSON result file path (optional)

        Returns:
            dict: Dictionary containing per-file inference details and summary statistics
        """
        if os.path.isfile(input_path):
            csv_files = [input_path]
        else:
            csv_files = sorted(glob.glob(os.path.join(input_path, '**', '*.csv'), recursive=True))

        if not csv_files:
            raise ValueError(f"No CSV files found: {input_path}")

        results = {}
        for csv_path in csv_files:
            filename = os.path.basename(csv_path)
            try:
                coords = load_and_normalize_csv(csv_path)
                result = self._segment_and_predict(coords)
                results[filename] = result
            except Exception as e:
                results[filename] = {'status': 'error', 'error': str(e)}

        ok_results = [r for r in results.values() if r.get('status') == 'ok']
        normal_count = sum(1 for r in ok_results if r['predicted_label'] == 'normal')
        abnormal_count = sum(1 for r in ok_results if r['predicted_label'] == 'abnormal')

        summary = {
            'total_processed': len(ok_results),
            'normal_count': normal_count,
            'abnormal_count': abnormal_count,
            'skipped_count': sum(1 for r in results.values() if r.get('status') == 'skip'),
            'error_count': sum(1 for r in results.values() if r.get('status') == 'error')
        }

        final_output = {
            'results': results,
            'summary': summary
        }

        if output_json_path:
            output_dir = os.path.dirname(output_json_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(final_output, f, ensure_ascii=False, indent=2)

        return final_output
