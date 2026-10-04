"""
Rule-based classifier: converts per-frame observations into an activity state.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
from backend.core.config import State, SystemConfig
from backend.vision.bed_detector import BedDetector


@dataclass
class Observation:
    bbox: Optional[Tuple[int, int, int, int]]
    pose: Optional["object"]      # PoseResult or None
    bed_box: Tuple[int, int, int, int]
    frame_w: int
    frame_h: int
    movement_px: float
    confidence: float


class StateClassifier:
    def __init__(self, config: SystemConfig):
        self.cfg = config

    def classify(self, obs: Observation) -> Tuple[str, float]:
        if obs.bbox is None:
            return State.UNKNOWN, 0.30

        pose = obs.pose
        if pose is None or pose.visibility_mean < 0.30:
            return self._classify_no_pose(obs)

        torso = pose.torso_angle
        knee = pose.knee_angle
        on_bed = self._on_bed(obs)
        movement_ratio = self._movement_ratio(obs)

        # --- LYING (sleeping / resting in bed) ---
        if torso < self.cfg.lying_angle_threshold:
            if on_bed:
                return State.LYING_IN_BED, min(0.95, 0.75 + pose.visibility_mean * 0.2)
            overlap = BedDetector.overlap_ratio(obs.bbox, obs.bed_box)
            if overlap > 0.25:
                return State.LYING_IN_BED, 0.70
            return State.OUT_OF_BED, 0.50

        # --- SITTING ---
        sitting_posture = (
            torso < self.cfg.sitting_angle_threshold
            or (knee is not None and knee < 115.0)
        )

        if sitting_posture:
            if on_bed:
                return State.SITTING_ON_BED, 0.85
            return State.SITTING_OUTSIDE_BED, 0.80

        # --- UPRIGHT: standing vs walking ---
        if movement_ratio > self.cfg.walking_movement_ratio and torso >= self.cfg.lying_angle_threshold:
            return State.WALKING, 0.82

        if on_bed:
            return State.SITTING_ON_BED, 0.60

        return State.STANDING, 0.80

    def _on_bed(self, obs: Observation) -> bool:
        bed = obs.bed_box
        bbox = obs.bbox
        if bbox is None:
            return False

        overlap = BedDetector.overlap_ratio(bbox, bed)
        if overlap >= self.cfg.bed_iou_threshold:
            return True

        pose = obs.pose
        if pose is not None:
            hip_pt = self._norm_to_pixel(pose.hip_center, obs.frame_w, obs.frame_h)
            shoulder_pt = self._norm_to_pixel(pose.shoulder_center, obs.frame_w, obs.frame_h)
            hip_in = BedDetector.point_in_box(hip_pt, bed)
            shoulder_in = BedDetector.point_in_box(shoulder_pt, bed)
            if hip_in and shoulder_in:
                return True
            if hip_in and overlap > 0.08:
                return True

        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        if BedDetector.point_in_box((cx, cy), bed) and overlap > 0.12:
            return True

        return False

    def _movement_ratio(self, obs: Observation) -> float:
        if obs.bbox is None:
            return 0.0
        bh = max(20, obs.bbox[3] - obs.bbox[1])
        return obs.movement_px / bh

    def _classify_no_pose(self, obs: Observation) -> Tuple[str, float]:
        bb = obs.bbox
        w = bb[2] - bb[0]
        h = bb[3] - bb[1]
        aspect = w / (h + 1e-6)
        on_bed = self._on_bed(obs)
        movement_ratio = self._movement_ratio(obs)

        if aspect > 1.25 and on_bed:
            return State.LYING_IN_BED, 0.60
        if aspect > 1.25 and not on_bed:
            return State.OUT_OF_BED, 0.45
        if on_bed:
            if aspect > 0.95:
                return State.LYING_IN_BED, 0.50
            return State.SITTING_ON_BED, 0.55
        if movement_ratio > self.cfg.walking_movement_ratio:
            return State.WALKING, 0.55
        return State.STANDING, 0.45

    @staticmethod
    def _norm_to_pixel(norm_pt, frame_w: int, frame_h: int):
        if norm_pt is None:
            return None
        return (norm_pt[0] * frame_w, norm_pt[1] * frame_h)
