"""
Bed region detection.

Strategy (in order):
  1. Manual region from UI/CLI → use as-is.
  2. YOLO COCO class 59 (bed) on calibration frames.
  3. Pose-based cluster when the person is lying (hip/shoulder centers).
  4. Expanded person bbox when horizontal posture is detected.
  5. Central-lower heuristic only as last resort.
"""
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.vision.person_detector import PersonDetector

BED_CLS = 59


class BedDetector:
    def __init__(
        self,
        manual_region: Optional[List[int]] = None,
        person_detector: Optional["PersonDetector"] = None,
    ):
        self.manual_region = manual_region
        self.person_detector = person_detector
        self._locked: Optional[Tuple[int, int, int, int]] = None
        self._yolo_boxes: List[Tuple[int, int, int, int]] = []
        self._lying_points: List[Tuple[float, float]] = []
        self._lying_bboxes: List[Tuple[int, int, int, int]] = []
        self._calib_count = 0
        self.max_calib_frames = 35

    def is_locked(self) -> bool:
        return self._locked is not None or (
            self.manual_region is not None and len(self.manual_region) == 4
        )

    def calibrate(
        self,
        frame_bgr,
        person_bbox: Optional[Tuple[int, int, int, int]] = None,
        pose=None,
    ) -> None:
        """Collect evidence from one frame before the main analysis loop."""
        if self.is_locked():
            return

        h, w = frame_bgr.shape[:2]

        if self.person_detector is not None:
            for bed_box in self.person_detector.detect_beds(frame_bgr, conf=0.20):
                self._yolo_boxes.append(bed_box)

        if pose is not None and pose.hip_center and pose.torso_angle < 50.0:
            hx = pose.hip_center[0] * w
            hy = pose.hip_center[1] * h
            self._lying_points.append((hx, hy))
            if pose.shoulder_center:
                sx = pose.shoulder_center[0] * w
                sy = pose.shoulder_center[1] * h
                self._lying_points.append((sx, sy))

        if person_bbox is not None:
            x1, y1, x2, y2 = person_bbox
            aspect = (x2 - x1) / (y2 - y1 + 1e-6)
            if pose is not None and pose.torso_angle < 50.0:
                self._lying_bboxes.append(person_bbox)
            elif aspect > 1.15:
                self._lying_bboxes.append(person_bbox)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                self._lying_points.append((cx, cy))

        self._calib_count += 1
        if self._calib_count >= self.max_calib_frames:
            self._lock_region(w, h)

    def finalize(self, width: int, height: int) -> None:
        if not self.is_locked():
            self._lock_region(width, height)

    def _lock_region(self, w: int, h: int) -> None:
        if self.manual_region and len(self.manual_region) == 4:
            self._locked = tuple(int(v) for v in self.manual_region)
            return

        candidates: List[Tuple[int, int, int, int]] = []

        if self._yolo_boxes:
            candidates.append(self._merge_boxes(self._yolo_boxes))

        if self._lying_bboxes:
            candidates.append(self._merge_boxes(self._lying_bboxes, pad_ratio=0.35))

        if len(self._lying_points) >= 2:
            xs = [p[0] for p in self._lying_points]
            ys = [p[1] for p in self._lying_points]
            pad_x = max(40.0, (max(xs) - min(xs)) * 0.6 + 80)
            pad_y = max(40.0, (max(ys) - min(ys)) * 0.6 + 60)
            x1 = int(max(0, min(xs) - pad_x))
            y1 = int(max(0, min(ys) - pad_y))
            x2 = int(min(w - 1, max(xs) + pad_x))
            y2 = int(min(h - 1, max(ys) + pad_y))
            if x2 - x1 > 50 and y2 - y1 > 50:
                candidates.append((x1, y1, x2, y2))

        if candidates:
            # Prefer YOLO if available, else largest area
            if self._yolo_boxes:
                self._locked = self._merge_boxes(self._yolo_boxes, pad_ratio=0.05)
            else:
                self._locked = max(candidates, key=self._area)
        else:
            self._locked = self._default_heuristic(w, h)

        self._locked = self._clip_box(self._locked, w, h)

    def detect(self, frame_shape, frame_bgr=None) -> Tuple[int, int, int, int]:
        if self.manual_region and len(self.manual_region) == 4:
            return tuple(int(v) for v in self.manual_region)

        if self._locked is not None:
            return self._locked

        if frame_bgr is not None:
            self.calibrate(frame_bgr)
            h, w = frame_bgr.shape[:2]
            if self._calib_count >= self.max_calib_frames:
                self.finalize(w, h)
            if self._locked is not None:
                return self._locked

        h, w = frame_shape[:2]
        return self._default_heuristic(w, h)

    @staticmethod
    def _default_heuristic(w: int, h: int) -> Tuple[int, int, int, int]:
        return (int(w * 0.10), int(h * 0.40), int(w * 0.90), int(h * 0.98))

    @staticmethod
    def _area(box) -> int:
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    @staticmethod
    def _clip_box(box, w: int, h: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        return (
            max(0, min(x1, w - 2)),
            max(0, min(y1, h - 2)),
            max(1, min(x2, w)),
            max(1, min(y2, h)),
        )

    @staticmethod
    def _merge_boxes(boxes: List[Tuple[int, int, int, int]], pad_ratio: float = 0.0):
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
        if pad_ratio > 0:
            bw, bh = x2 - x1, y2 - y1
            x1 -= int(bw * pad_ratio)
            y1 -= int(bh * pad_ratio)
            x2 += int(bw * pad_ratio)
            y2 += int(bh * pad_ratio)
        return (x1, y1, x2, y2)

    @staticmethod
    def iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / ua if ua > 0 else 0.0

    @staticmethod
    def point_in_box(pt, box) -> bool:
        if pt is None or box is None:
            return False
        x, y = pt
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    @staticmethod
    def overlap_ratio(person_box, bed_box) -> float:
        """Fraction of person bbox area inside bed box."""
        if person_box is None or bed_box is None:
            return 0.0
        px1, py1, px2, py2 = person_box
        bx1, by1, bx2, by2 = bed_box
        ix1, iy1 = max(px1, bx1), max(py1, by1)
        ix2, iy2 = min(px2, bx2), min(py2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        person_area = max(1, (px2 - px1) * (py2 - py1))
        return inter / person_area
