"""MediaPipe Pose estimation — extracts 33 body landmarks (Tasks API)."""
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import math
import os


@dataclass
class PoseResult:
    landmarks: Dict[int, Tuple[float, float, float]]  # id → (x_norm, y_norm, visibility)
    torso_angle: float          # degrees from horizontal (0 = flat, 90 = upright)
    hip_center: Optional[Tuple[float, float]]  # normalized (x, y)
    shoulder_center: Optional[Tuple[float, float]]
    knee_angle: Optional[float] # degrees
    visibility_mean: float


# Default model path — auto-downloads if missing
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "pose_landmarker_lite.task")


def _ensure_model():
    """Download the pose landmarker model if not already present."""
    if os.path.isfile(_MODEL_PATH):
        return _MODEL_PATH
    import urllib.request
    print(f"⬇ Downloading pose_landmarker_lite.task ...")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print(f"✔ Downloaded to {_MODEL_PATH}")
    return _MODEL_PATH


class PoseEstimator:
    def __init__(self, model_complexity: int = 1):
        import mediapipe as mp

        model_path = _ensure_model()

        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._mp_image_cls = mp.Image
        self._image_format_srgb = mp.ImageFormat.SRGB

        from backend.core.constants import MP
        self.MP = MP

    def estimate(self, frame_bgr, person_bbox=None) -> Optional[PoseResult]:
        import numpy as np
        import cv2

        # Crop to person bbox for better accuracy (optional)
        h, w = frame_bgr.shape[:2]
        if person_bbox is not None:
            x1, y1, x2, y2 = person_bbox
            x1 = max(0, int(x1)); y1 = max(0, int(y1))
            x2 = min(w, int(x2)); y2 = min(h, int(y2))
            if x2 - x1 < 10 or y2 - y1 < 10:
                crop = frame_bgr
                offset = (0, 0)
            else:
                crop = frame_bgr[y1:y2, x1:x2]
                offset = (x1, y1)
        else:
            crop = frame_bgr
            offset = (0, 0)

        # Convert BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = self._mp_image_cls(image_format=self._image_format_srgb, data=rgb)

        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        pose_landmarks = result.pose_landmarks[0]  # first (and only) pose

        ch, cw = crop.shape[:2]
        ox, oy = offset
        lm: Dict[int, Tuple[float, float, float]] = {}
        for i, p in enumerate(pose_landmarks):
            # p.x, p.y are normalized [0,1] within the crop
            gx = (p.x * cw + ox) / w
            gy = (p.y * ch + oy) / h
            vis = p.visibility if p.visibility is not None else 0.0
            lm[i] = (gx, gy, vis)

        # Torso angle from horizontal
        ls = lm.get(self.MP.LEFT_SHOULDER)
        rs = lm.get(self.MP.RIGHT_SHOULDER)
        lh = lm.get(self.MP.LEFT_HIP)
        rh = lm.get(self.MP.RIGHT_HIP)

        torso_angle = 90.0
        shoulder_center = None
        hip_center = None
        if ls and rs and lh and rh:
            shoulder_center = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
            hip_center = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
            dy = hip_center[1] - shoulder_center[1]
            dx = hip_center[0] - shoulder_center[0]
            torso_angle = abs(math.degrees(math.atan2(abs(dy), abs(dx) + 1e-6)))

        # Knee angle (left)
        knee_angle = None
        lk = lm.get(self.MP.LEFT_KNEE)
        la = lm.get(self.MP.LEFT_ANKLE)
        if lh and lk and la:
            v1 = (lh[0] - lk[0], lh[1] - lk[1])
            v2 = (la[0] - lk[0], la[1] - lk[1])
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            n1 = math.hypot(*v1) + 1e-6
            n2 = math.hypot(*v2) + 1e-6
            cos_a = max(-1.0, min(1.0, dot / (n1 * n2)))
            knee_angle = math.degrees(math.acos(cos_a))

        vis_mean = sum(v[2] for v in lm.values()) / max(1, len(lm))

        return PoseResult(
            landmarks=lm,
            torso_angle=torso_angle,
            hip_center=hip_center,
            shoulder_center=shoulder_center,
            knee_angle=knee_angle,
            visibility_mean=vis_mean,
        )

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            pass