import os
import base64
import tempfile

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QStyle
from PySide6.QtCore import Qt, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget

TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "temp")


class VideoPlayer(QWidget):
    def __init__(self, clip_b64: str, parent=None):
        super().__init__(parent)
        self._tmp_file: str | None = None
        self._setup_ui()
        self._load_clip(clip_b64)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._video_widget = QVideoWidget()
        self._video_widget.setMinimumHeight(180)
        layout.addWidget(self._video_widget, 1)

        controls = QHBoxLayout()
        controls.setContentsMargins(4, 0, 4, 0)

        self._btn_play = QPushButton()
        self._btn_play.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self._btn_play.setFixedSize(32, 28)
        self._btn_play.clicked.connect(self._toggle_play)
        controls.addWidget(self._btn_play)

        self._slider = QHBoxLayout()
        from PySide6.QtWidgets import QSlider
        self._progress_slider = QSlider(Qt.Orientation.Horizontal)
        self._progress_slider.setRange(0, 1000)
        self._progress_slider.sliderMoved.connect(self._seek)
        controls.addWidget(self._progress_slider, 1)

        self._lbl_time = QLabel("0:00")
        self._lbl_time.setFixedWidth(40)
        controls.addWidget(self._lbl_time)

        layout.addLayout(controls)

        self._player = QMediaPlayer()
        self._audio = QAudioOutput()
        self._audio.setMuted(True)
        self._player.setAudioOutput(self._audio)
        self._player.setVideoOutput(self._video_widget)
        self._player.positionChanged.connect(self._on_position)
        self._player.durationChanged.connect(self._on_duration)

    def _load_clip(self, clip_b64: str):
        try:
            data = base64.b64decode(clip_b64)
            fd, path = tempfile.mkstemp(suffix='.mp4', dir=TEMP_DIR)
            os.write(fd, data)
            os.close(fd)
            self._tmp_file = path
            self._player.setSource(QUrl.fromLocalFile(path))
        except Exception:
            pass

    def _toggle_play(self):
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
            self._btn_play.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        else:
            self._player.play()
            self._btn_play.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))

    def _seek(self, pos: int):
        duration = self._player.duration()
        if duration > 0:
            self._player.setPosition(int(pos * duration / 1000))

    def _on_position(self, pos: int):
        duration = self._player.duration()
        if duration > 0:
            self._progress_slider.blockSignals(True)
            self._progress_slider.setValue(int(pos * 1000 / duration))
            self._progress_slider.blockSignals(False)
        secs = pos // 1000
        self._lbl_time.setText(f"{secs // 60}:{secs % 60:02d}")

    def _on_duration(self, duration: int):
        self._progress_slider.setRange(0, 1000)

    def stop(self):
        self._player.stop()

    def cleanup(self):
        self._player.stop()
        if self._tmp_file and os.path.exists(self._tmp_file):
            try:
                os.unlink(self._tmp_file)
            except OSError:
                pass
