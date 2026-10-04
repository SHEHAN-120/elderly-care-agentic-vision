"""Frame sampling helper — converts frame index ↔ seconds."""
from dataclasses import dataclass


@dataclass
class Sample:
    frame_index: int
    time_sec: float
    frame: "object"  # np.ndarray


class FrameSampler:
    def __init__(self, fps: float, interval: int = 15):
        self.fps = fps
        self.interval = max(1, interval)

    def time_of(self, frame_index: int) -> float:
        return frame_index / self.fps if self.fps > 0 else 0.0

    def sample(self, frame_index: int, frame) -> Sample:
        return Sample(
            frame_index=frame_index,
            time_sec=self.time_of(frame_index),
            frame=frame,
        )