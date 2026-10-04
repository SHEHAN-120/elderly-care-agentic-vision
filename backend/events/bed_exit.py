"""Bed exit sequence detection."""
from typing import List, Optional
from backend.core.config import State
from backend.state.state_tracker import StateSegment


class BedExitDetector:
    """
    Sequence: in-bed → sitting_on_bed → standing/walking → moving away
    Emits a bed_exit event when the sequence is confirmed.
    """

    def __init__(self):
        self.armed = False
        self._start_time: Optional[float] = None
        self._prev_state: Optional[str] = None

    def observe(self, seg: StateSegment) -> Optional[dict]:
        s = seg.state

        if s in (State.LYING_IN_BED, State.SITTING_ON_BED):
            self.armed = True
            self._start_time = seg.end_sec
            self._prev_state = s
            return None

        if not self.armed:
            return None

        # Confirmed exit: upright / away from bed for a sustained segment
        if s in (State.STANDING, State.WALKING, State.OUT_OF_BED, State.SITTING_OUTSIDE_BED):
            if seg.duration() >= 1.2:
                if self._prev_state == State.LYING_IN_BED and s == State.STANDING and seg.duration() < 2.5:
                    return None
                event = {
                    "event": "bed_exit",
                    "start_time": self._fmt(self._start_time or seg.start_sec),
                    "confirmed_time": self._fmt(seg.end_sec),
                    "previous_state": self._prev_state,
                    "current_state": s,
                    "confidence": round(seg.confidence, 2),
                    "decision": "MONITOR",
                }
                self.armed = False
                return event
        return None

    @staticmethod
    def _fmt(t: float) -> str:
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"