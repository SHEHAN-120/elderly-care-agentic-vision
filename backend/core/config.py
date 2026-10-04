"""
System configuration: activity states, thresholds, alert rules.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Set


# ---------------------------------------------------------------------------
# Activity States
# ---------------------------------------------------------------------------
class State:
    LYING_IN_BED = "LYING_IN_BED"
    SITTING_ON_BED = "SITTING_ON_BED"
    SITTING_OUTSIDE_BED = "SITTING_OUTSIDE_BED"
    STANDING = "STANDING"
    WALKING = "WALKING"
    OUT_OF_BED = "OUT_OF_BED"
    UNKNOWN = "UNKNOWN"


ALL_STATES: List[str] = [
    State.LYING_IN_BED,
    State.SITTING_ON_BED,
    State.SITTING_OUTSIDE_BED,
    State.STANDING,
    State.WALKING,
    State.OUT_OF_BED,
    State.UNKNOWN,
]

IN_BED_STATES: Set[str] = {State.LYING_IN_BED, State.SITTING_ON_BED}

# Allowed transitions (used by StateTracker to reject invalid jumps)
VALID_TRANSITIONS: Dict[str, Set[str]] = {
    State.LYING_IN_BED: {State.SITTING_ON_BED, State.LYING_IN_BED, State.UNKNOWN},
    State.SITTING_ON_BED: {State.LYING_IN_BED, State.STANDING, State.SITTING_ON_BED, State.UNKNOWN},
    State.STANDING: {State.SITTING_ON_BED, State.WALKING, State.SITTING_OUTSIDE_BED, State.STANDING, State.UNKNOWN},
    State.WALKING: {State.STANDING, State.SITTING_OUTSIDE_BED, State.OUT_OF_BED, State.WALKING, State.UNKNOWN},
    State.SITTING_OUTSIDE_BED: {State.STANDING, State.WALKING, State.SITTING_OUTSIDE_BED, State.UNKNOWN},
    State.OUT_OF_BED: {State.WALKING, State.STANDING, State.UNKNOWN, State.OUT_OF_BED},
    State.UNKNOWN: set(ALL_STATES),
}


# ---------------------------------------------------------------------------
# System Config
# ---------------------------------------------------------------------------
@dataclass
class SystemConfig:
    # --- Video ---
    frame_sample_interval: int = 15          # process every Nth frame
    max_processed_frames: int = 5000         # safety cap
    bed_calib_frames: int = 35               # frames to auto-detect bed region

    # --- Pose / Body posture thresholds (degrees) ---
    lying_angle_threshold: float = 45.0      # torso angle from horizontal
    sitting_angle_threshold: float = 65.0    # torso angle for sitting
    standing_min_torso_angle: float = 60.0

    # --- Movement (movement_px / person_bbox_height per sampled frame) ---
    walking_movement_ratio: float = 0.06
    walking_movement_threshold: float = 15.0  # legacy px threshold (unused in classifier)
    standing_movement_threshold: float = 5.0

    # --- Bed region ---
    bed_iou_threshold: float = 0.12           # min person-in-bed overlap ratio
    bed_center_in_bed: bool = True            # require hip center inside bed box

    # --- Temporal smoothing ---
    state_stability_frames: int = 4           # frames to confirm a new state
    min_state_duration_sec: float = 1.0

    # --- Agentic reasoning ---
    agent_confidence_threshold: float = 0.60  # below → request context
    agent_context_window_sec: float = 5.0

    # --- Alert thresholds ---
    prolonged_out_of_bed_sec: float = 300.0   # 5 min  → ALERT
    prolonged_edge_sitting_sec: float = 120.0 # 2 min  → MONITOR
    prolonged_unknown_sec: float = 30.0       # 30 sec → MONITOR

    # --- Output ---
    save_annotated: bool = False
    output_dir: str = "outputs"

    # --- Detection ---
    person_conf_threshold: float = 0.35
    person_model: str = "yolov8n.pt"          # auto-downloaded

    # --- Optional manual bed region (x1,y1,x2,y2) ---
    manual_bed_region: List[int] = field(default_factory=list)


DEFAULT_CONFIG = SystemConfig()