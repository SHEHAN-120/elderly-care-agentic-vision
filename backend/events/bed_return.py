"""Return-to-bed sequence detection."""
from typing import Optional
from backend.core.config import State
from backend.state.state_tracker import StateSegment


class BedReturnDetector:
    """
    Sequence: out_of_bed → approaches → sitting_on_bed → lying_in_bed
    Emits bed_return when a new in-bed segment follows an out-of-bed period.
    """

    def __init__(self):
        self._left_bed = False
        self._start_time: Optional[float] = None
        self._prev_state: Optional[str] = None

    def observe(self, seg: StateSegment) -> Optional[dict]:
        s = seg.state

        if s in (State.STANDING, State.WALKING, State.OUT_OF_BED, State.SITTING_OUTSIDE_BED):
            self._left_bed = True
            self._start_time = seg.end_sec
            self._prev_state = s
            return None

        if self._left_bed and s == State.SITTING_ON_BED:
            self._prev_state = s
            return None

        if self._left_bed and s == State.LYING_IN_BED:
            event = {
                "event": "bed_return",
                "start_time": self._fmt(self._start_time or seg.start_sec),
                "confirmed_time": self._fmt(seg.end_sec),
                "previous_state": self._prev_state,
                "current_state": s,
                "confidence": round(seg.confidence, 2),
                "decision": "NORMAL",
            }
            self._left_bed = False
            return event
        return None

    @staticmethod
    def _fmt(t: float) -> str:
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"