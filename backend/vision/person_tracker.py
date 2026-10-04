"""Lightweight SORT-style tracker (Kalman filter + Hungarian)."""
from typing import List, Tuple, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / ua if ua > 0 else 0.0


def bbox_to_z(bbox):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w/2, y1 + h/2
    s = w * h
    r = w / (h + 1e-6)
    return np.array([[cx], [cy], [s], [r]], dtype=float)


def z_to_bbox(z):
    cx, cy, s, r = z[0, 0], z[1, 0], z[2, 0], z[3, 0]
    w = np.sqrt(max(s, 1e-6) * max(r, 1e-6))
    h = max(s, 1e-6) / (w + 1e-6)
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]


class KalmanTrack:
    _next_id = 1

    def __init__(self, bbox):
        self.id = KalmanTrack._next_id
        KalmanTrack._next_id += 1
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.eye(7)
        for i in range(3):
            self.kf.F[i, i+4] = 1
        self.kf.H = np.eye(4, 7)
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = bbox_to_z(bbox)
        self.time_since_update = 0
        self.hits = 1

    def predict(self):
        if self.kf.x[6, 0] + self.kf.x[2, 0] <= 0:
            self.kf.x[6, 0] = 0.0
        self.kf.predict()
        self.time_since_update += 1
        return z_to_bbox(self.kf.x)

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.kf.update(bbox_to_z(bbox))

    def bbox(self):
        return z_to_bbox(self.kf.x)


class PersonTracker:
    def __init__(self, max_age: int = 5, min_hits: int = 1, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[KalmanTrack] = []

    def update(self, detections: List[Tuple[int, int, int, int]]) -> List[Tuple[int, List[int]]]:
        """Returns list of (track_id, bbox)."""
        # Predict
        for t in self.tracks:
            t.predict()

        if len(self.tracks) == 0:
            for d in detections:
                self.tracks.append(KalmanTrack(d))
            return [(t.id, list(t.bbox())) for t in self.tracks]

        # Association matrix
        cost = np.zeros((len(self.tracks), len(detections)))
        for i, t in enumerate(self.tracks):
            tb = t.bbox()
            for j, d in enumerate(detections):
                cost[i, j] = 1.0 - iou(tb, d)

        row, col = linear_sum_assignment(cost)

        matched_t, matched_d = set(), set()
        for r, c in zip(row, col):
            if cost[r, c] > 1.0 - self.iou_threshold:
                continue
            self.tracks[r].update(detections[c])
            matched_t.add(r); matched_d.add(c)

        # Unmatched detections → new tracks
        for j, d in enumerate(detections):
            if j not in matched_d:
                self.tracks.append(KalmanTrack(d))

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        out = []
        for t in self.tracks:
            if t.time_since_update == 0 and t.hits >= self.min_hits:
                out.append((t.id, list(t.bbox())))
        return out