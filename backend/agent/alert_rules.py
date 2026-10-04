"""
Contextual alert rules → NORMAL / MONITOR / ALERT.
"""
from typing import List, Optional
from backend.core.config import State, SystemConfig
from backend.state.state_tracker import StateSegment


class AlertRules:
    def __init__(self, config: SystemConfig):
        self.cfg = config

    def evaluate(
        self,
        segments: List[StateSegment],
        final_state: str,
        longest_out_of_bed_sec: float,
        current_time_sec: float,
    ) -> dict:
        # Rule 1: Prolonged out-of-bed
        if longest_out_of_bed_sec > self.cfg.prolonged_out_of_bed_sec:
            return {
                "decision": "ALERT",
                "reason": f"Prolonged out-of-bed period ({longest_out_of_bed_sec:.0f}s > "
                          f"{self.cfg.prolonged_out_of_bed_sec:.0f}s). Possible wandering / fall.",
            }

        # Rule 2: Extended edge sitting (last segment)
        if segments:
            last = segments[-1]
            if last.state == State.SITTING_ON_BED and last.duration() > self.cfg.prolonged_edge_sitting_sec:
                return {
                    "decision": "MONITOR",
                    "reason": f"Extended edge-of-bed sitting ({last.duration():.0f}s). "
                              f"Possible dizziness/confusion.",
                }

        # Rule 3: Prolonged unknown
        if segments:
            last = segments[-1]
            if last.state == State.UNKNOWN and last.duration() > self.cfg.prolonged_unknown_sec:
                return {
                    "decision": "MONITOR",
                    "reason": f"Extended unknown state ({last.duration():.0f}s). Low confidence.",
                }

        # Rule 4: Current out-of-bed for a long time
        if final_state in (State.OUT_OF_BED, State.WALKING):
            # find last continuous out-of-bed streak
            streak = 0.0
            for seg in reversed(segments):
                if seg.state not in (State.LYING_IN_BED, State.SITTING_ON_BED):
                    streak += seg.duration()
                else:
                    break
            if streak > self.cfg.prolonged_out_of_bed_sec * 0.7:
                return {
                    "decision": "MONITOR",
                    "reason": f"Currently out of bed for {streak:.0f}s — approaching alert threshold.",
                }

        return {
            "decision": "NORMAL",
            "reason": f"Person is in state {final_state} — normal activity.",
        }