"""Human-readable timeline generation."""
from typing import List
from backend.state.state_tracker import StateSegment


def fmt(t: float) -> str:
    m = int(t // 60)
    s = int(t % 60)
    return f"{m:02d}:{s:02d}"


def timeline_text(segments: List[StateSegment]) -> str:
    lines = []
    for seg in segments:
        if seg.duration() < 0.01:
            continue
        lines.append(f"{fmt(seg.start_sec)} – {fmt(seg.end_sec)}  {seg.state}")
    return "\n".join(lines)


def timeline_json(segments: List[StateSegment]) -> List[dict]:
    return [seg.to_dict() for seg in segments if seg.duration() > 0.01]