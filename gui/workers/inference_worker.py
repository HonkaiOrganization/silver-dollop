from PySide6.QtCore import QThread, Signal

from core.infer import JumpRopeInference


class InferenceWorker(QThread):
    finished = Signal(dict)

    def __init__(self, inference: JumpRopeInference, csv_path: str, output_json: str):
        super().__init__()
        self._inference = inference
        self._csv_path = csv_path
        self._output_json = output_json
        self._abandoned = False

    def abandon(self):
        self._abandoned = True

    def run(self):
        result = self._inference.predict(self._csv_path, self._output_json)
        if not self._abandoned:
            self.finished.emit(result)
