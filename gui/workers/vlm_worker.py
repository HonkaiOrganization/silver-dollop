from PySide6.QtCore import QThread, Signal

from core.vlm import analyze_windows


class VLMWorker(QThread):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, video_path: str, json_path: str, top_k: int = 3):
        super().__init__()
        self.video_path = video_path
        self.json_path = json_path
        self.top_k = top_k

    def run(self):
        try:
            result = analyze_windows(
                self.video_path, self.json_path, top_k=self.top_k,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
