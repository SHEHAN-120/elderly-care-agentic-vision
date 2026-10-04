"""Coordinates bed-exit and bed-return detection over the segment timeline."""
from typing import List
from backend.events.bed_exit import BedExitDetector
from backend.events.bed_return import BedReturnDetector
from backend.state.state_tracker import StateSegment


class EventManager:
    def __init__(self):
        self.exit_det = BedExitDetector()
        self.return_det = BedReturnDetector()
        self.events: List[dict] = []

    def process(self, segments: List[StateSegment]):
        for seg in segments:
            e = self.exit_det.observe(seg)
            if e:
                self.events.append(e)
            r = self.return_det.observe(seg)
            if r:
                self.events.append(r)
        return self.events

    def count(self, kind: str) -> int:
        return sum(1 for e in self.events if e["event"] == kind)