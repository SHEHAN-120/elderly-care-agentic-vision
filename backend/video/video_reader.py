"""Video I/O: metadata + frame iteration."""
import cv2
from dataclasses import dataclass
from typing import Iterator, Tuple


@dataclass
class VideoMeta:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_sec(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


class VideoReader:
    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        self.meta = VideoMeta(
            path=path,
            fps=self.cap.get(cv2.CAP_PROP_FPS) or 25.0,
            frame_count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def frames(self, step: int = 1) -> Iterator[Tuple[int, "cv2.Mat"]]:
        """Yield (frame_index, frame_bgr) every `step` frames."""
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            if idx % step == 0:
                yield idx, frame
            idx += 1

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None