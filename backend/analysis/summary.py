"""Builds the complete JSON output (matches assignment spec)."""
import json
from typing import List
from backend.core.config import State
from backend.state.state_tracker import StateSegment
from backend.analysis.duration import activity_durations, bed_summary
from backend.analysis.timeline import timeline_text, timeline_json


def build_summary(
    observation_duration_sec: float,
    segments: List[StateSegment],
    events: List[dict],
    alert: dict,
) -> dict:
    acts = activity_durations(segments)
    bed = bed_summary(segments)
    final_state = segments[-1].state if segments else State.UNKNOWN

    return {
        "observation_duration_sec": round(observation_duration_sec, 2),
        "activity_duration_sec": {k: round(v, 2) for k, v in acts.items()},
        "bed_exit_count": sum(1 for e in events if e["event"] == "bed_exit"),
        "bed_return_count": sum(1 for e in events if e["event"] == "bed_return"),
        "total_in_bed_sec": bed["total_in_bed_sec"],
        "total_out_of_bed_sec": bed["total_out_of_bed_sec"],
        "longest_out_of_bed_period_sec": bed["longest_out_of_bed_period_sec"],
        "final_state": final_state,
        "events": events,
        "current_alert": alert,
    }


def summary_to_json(summary: dict) -> str:
    return json.dumps(summary, indent=2)