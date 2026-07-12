from gui.widgets.frame_display import FrameDisplay


class VideoDisplay(FrameDisplay):
    """Displays the raw camera / playback video feed."""

    def __init__(self, parent=None):
        super().__init__(title="Camera Feed", parent=parent)


class SkeletonDisplay(FrameDisplay):
    """Displays the pose skeleton rendering."""

    def __init__(self, parent=None):
        super().__init__(title="Pose Skeleton", parent=parent)
