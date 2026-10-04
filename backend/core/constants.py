"""Shared constants: MediaPipe landmark indices, colors, labels."""

# MediaPipe Pose landmark indices (33 points)
class MP:
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


# Annotation colors (BGR)
COLOR_GREEN  = (0, 200, 0)
COLOR_YELLOW = (0, 200, 200)
COLOR_RED    = (0, 0, 220)
COLOR_BLUE   = (220, 120, 0)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0, 0, 0)
COLOR_BED    = (180, 100, 255)


# Map state → annotation color
STATE_COLORS = {
    "LYING_IN_BED": COLOR_GREEN,
    "SITTING_ON_BED": COLOR_YELLOW,
    "SITTING_OUTSIDE_BED": COLOR_BLUE,
    "STANDING": COLOR_GREEN,
    "WALKING": COLOR_BLUE,
    "OUT_OF_BED": COLOR_RED,
    "UNKNOWN": COLOR_WHITE,
}