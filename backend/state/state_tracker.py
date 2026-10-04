"""
Temporal smoothing + transition validation.
Only confirmed states are emitted downstream.
"""
from collections import deque
from typing import Deque, List, Optional, Tuple
from backend.core.config import State, SystemConfig, VALID_TRANSITIONS


class StateSegment:
    def __init__(self, state: str, start_sec: float, confidence: float):
        self.state = state
        self.start_sec = start_sec
        self.end_sec = start_sec
        self.confidence = confidence

    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def to_dict(self):
        return {
            "state": self.state,
            "start_sec": round(self.start_sec, 2),
            "end_sec": round(self.end_sec, 2),
            "duration_sec": round(self.duration(), 2),
            "confidence": round(self.confidence, 3),
        }


class StateTracker:
    def __init__(self, config: SystemConfig):
        self.cfg = config
        self.buffer: Deque[Tuple[str, float]] = deque(maxlen=config.state_stability_frames)
        self.current_state: Optional[str] = None
        self.current_conf: float = 0.0
        self.segments: List[StateSegment] = []
        self._seg_start: float = 0.0
        self._seg_conf_sum: float = 0.0
        self._seg_conf_n: int = 0

    def update(self, raw_state: str, confidence: float, time_sec: float) -> Optional[str]:
        """Feed a raw frame classification. Returns confirmed state if changed."""
        self.buffer.append((raw_state, confidence))

        if len(self.buffer) < self.cfg.state_stability_frames:
            return None

        # Majority vote
        votes = {}
        confs = {}
        for s, c in self.buffer:
            votes[s] = votes.get(s, 0) + 1
            confs[s] = confs.get(s, 0.0) + c
        winner = max(votes.items(), key=lambda kv: (kv[1], confs[kv[0]]))[0]
        avg_conf = confs[winner] / votes[winner]

        # Avoid flickering into UNKNOWN from a stable state on a single noisy frame
        if (
            self.current_state is not None
            and self.current_state != State.UNKNOWN
            and winner == State.UNKNOWN
            and votes.get(State.UNKNOWN, 0) < self.cfg.state_stability_frames
        ):
            winner = self.current_state
            avg_conf = confs.get(winner, avg_conf) / max(1, votes.get(winner, 1))

        if self.current_state is None:
            self._start_new(winner, avg_conf, time_sec)
            return self.current_state

        if winner == self.current_state:
            self._seg_conf_sum += avg_conf
            self._seg_conf_n += 1
            self.segments[-1].end_sec = time_sec
            self.segments[-1].confidence = self._seg_conf_sum / max(1, self._seg_conf_n)
            return None

        # Validate transition — hold current state instead of jumping to UNKNOWN
        allowed = VALID_TRANSITIONS.get(self.current_state, set())
        if winner not in allowed and winner != self.current_state:
            winner = self.current_state
            avg_conf = self.current_conf

        if winner == self.current_state:
            return None

        self._start_new(winner, avg_conf, time_sec)
        return self.current_state

    def _start_new(self, state: str, conf: float, time_sec: float):
        if self.current_state is not None and self.segments:
            self.segments[-1].end_sec = time_sec
        self.current_state = state
        self.current_conf = conf
        self._seg_start = time_sec
        self._seg_conf_sum = conf
        self._seg_conf_n = 1
        self.segments.append(StateSegment(state, time_sec, conf))

    def finalize(self, end_time_sec: float):
        if self.segments:
            self.segments[-1].end_sec = end_time_sec

    def timeline(self) -> List[StateSegment]:
        return self.segments