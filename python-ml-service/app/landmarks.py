"""Stable landmark feature layout shared by collection and model inference."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any

LANDMARK_COUNT = 21
COORDINATES = ("x", "y", "z")
HANDEDNESS_ORDER = ("Left", "Right")


def hand_feature_names(side: str) -> list[str]:
    """Return presence and wrist-relative coordinate fields for one hand slot."""
    if side not in HANDEDNESS_ORDER:
        raise ValueError(f"Unknown hand side: {side}")
    fields = [f"{side.lower()}_present"]
    fields.extend(
        f"{side.lower()}_{axis}{index}"
        for index in range(LANDMARK_COUNT)
        for axis in COORDINATES
    )
    return fields


FEATURE_COLUMNS = hand_feature_names("Left") + hand_feature_names("Right")


def normalize_landmarks(landmarks: Sequence[Any]) -> list[float]:
    """Translate landmarks to the wrist and scale by the farthest 3D point.

    MediaPipe provides 21 ordered landmarks. Wrist-relative coordinates remove
    image position; scale normalization reduces sensitivity to camera distance.
    The hand's orientation is intentionally preserved because it carries sign
    meaning. Returns an empty list for invalid or degenerate input.
    """
    if len(landmarks) != LANDMARK_COUNT:
        return []

    points: list[tuple[float, float, float]] = []
    try:
        for landmark in landmarks:
            points.append((float(landmark.x), float(landmark.y), float(landmark.z)))
    except (AttributeError, TypeError, ValueError):
        return []
    if any(not isfinite(component) for point in points for component in point):
        return []

    origin = points[0]
    translated = [tuple(point[i] - origin[i] for i in range(3)) for point in points]
    scale = max(
        (sum(component * component for component in point) ** 0.5 for point in translated),
        default=0.0,
    )
    if scale <= 1e-8:
        return []

    return [component / scale for point in translated for component in point]


def frame_feature_values(
    multi_hand_landmarks: Sequence[Sequence[Any]],
    multi_handedness: Sequence[Sequence[Any]],
) -> dict[str, float] | None:
    """Create the fixed 128-value left/right layout for one detected frame.

    A missing hand is represented by an all-zero coordinate slot and presence
    value 0. Hands are put in handedness slots so their order does not flicker
    when MediaPipe changes detection order between frames.
    """
    if not multi_hand_landmarks:
        return None

    values = {column: 0.0 for column in FEATURE_COLUMNS}
    seen_sides: set[str] = set()
    for index, hand in enumerate(multi_hand_landmarks):
        if index >= len(multi_handedness) or not multi_handedness[index]:
            continue
        side = str(multi_handedness[index][0].classification[0].label)
        if side not in HANDEDNESS_ORDER or side in seen_sides:
            continue
        normalized = normalize_landmarks(hand.landmark)
        if not normalized:
            continue
        seen_sides.add(side)
        prefix = side.lower()
        values[f"{prefix}_present"] = 1.0
        for landmark_index in range(LANDMARK_COUNT):
            for axis_index, axis in enumerate(COORDINATES):
                name = f"{prefix}_{axis}{landmark_index}"
                values[name] = normalized[landmark_index * 3 + axis_index]

    if not seen_sides:
        return None
    return values
