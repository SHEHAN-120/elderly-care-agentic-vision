"""Activity duration + bed summary calculations."""
from typing import Dict, List
from backend.core.config import State, IN_BED_STATES
from backend.state.state_tracker import StateSegment


def activity_durations(segments: List[StateSegment]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for seg in segments:
        key = seg.state.lower()
        out[key] = out.get(key, 0.0) + seg.duration()
    return out


def bed_summary(segments: List[StateSegment]) -> Dict[str, float]:
    in_bed = 0.0
    out_bed = 0.0
    longest_out = 0.0
    streak = 0.0
    for seg in segments:
        if seg.state in IN_BED_STATES:
            in_bed += seg.duration()
            longest_out = max(longest_out, streak)
            streak = 0.0
        else:
            out_bed += seg.duration()
            streak += seg.duration()
    longest_out = max(longest_out, streak)
    return {
        "total_in_bed_sec": round(in_bed, 2),
        "total_out_of_bed_sec": round(out_bed, 2),
        "longest_out_of_bed_period_sec": round(longest_out, 2),
    }