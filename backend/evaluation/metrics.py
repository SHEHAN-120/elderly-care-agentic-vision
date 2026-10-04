"""
Evaluation metrics: activity accuracy, bed-exit precision/recall, duration error.
Ground truth JSON format:
{
  "duration_sec": 1200,
  "timeline": [{"start_sec":0,"end_sec":272,"state":"LYING_IN_BED"}, ...],
  "activity_duration_sec": {"lying_in_bed":702, ...},
  "bed_exit_count": 2
}
"""
import json
from typing import List, Dict


def load_gt(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _state_at(t: float, timeline: List[dict]) -> str:
    for seg in timeline:
        if seg["start_sec"] <= t < seg["end_sec"]:
            return seg["state"]
    return "UNKNOWN"


def frame_accuracy(gt_timeline: List[dict], pred_segments: List["object"], step: float = 1.0) -> float:
    if not gt_timeline or not pred_segments:
        return 0.0
    end = max(seg["end_sec"] for seg in gt_timeline)
    total, correct = 0, 0
    t = 0.0
    while t < end:
        gt = _state_at(t, gt_timeline)
        pr = "UNKNOWN"
        for s in pred_segments:
            if s.start_sec <= t < s.end_sec:
                pr = s.state
                break
        total += 1
        if gt == pr:
            correct += 1
        t += step
    return correct / max(1, total)


def bed_exit_pr(gt_count: int, pred_count: int) -> Dict[str, float]:
    # Simple count-based approximation (event matching requires timestamps)
    tp = min(gt_count, pred_count)
    precision = tp / pred_count if pred_count else 0.0
    recall = tp / gt_count if gt_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp,
            "fp": max(0, pred_count - gt_count), "fn": max(0, gt_count - pred_count)}


def duration_errors(gt_durations: Dict[str, float], pred_durations: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for k, gt in gt_durations.items():
        pr = pred_durations.get(k, 0.0)
        out[k] = round(pr - gt, 2)
    return out


def confusion_matrix(gt_timeline: List[dict], pred_segments: List["object"], states: List[str], step: float = 1.0):
    idx = {s: i for i, s in enumerate(states)}
    mat = [[0]*len(states) for _ in states]
    if not gt_timeline:
        return mat, states
    end = max(seg["end_sec"] for seg in gt_timeline)
    t = 0.0
    while t < end:
        gt = _state_at(t, gt_timeline)
        pr = "UNKNOWN"
        for s in pred_segments:
            if s.start_sec <= t < s.end_sec:
                pr = s.state
                break
        if gt in idx and pr in idx:
            mat[idx[gt]][idx[pr]] += 1
        t += step
    return mat, states