"""CSV storage helpers for labeled landmark samples."""

from __future__ import annotations

import csv
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from app.landmarks import FEATURE_COLUMNS

METADATA_COLUMNS = (
    "sample_id",
    "capture_session_id",
    "gesture_label",
    "captured_at_utc",
    "hand_count",
)
CSV_COLUMNS = [*METADATA_COLUMNS, *FEATURE_COLUMNS]
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def validate_label(label: str) -> str:
    """Restrict labels to safe, readable class identifiers."""
    normalized = label.strip()
    if not LABEL_PATTERN.fullmatch(normalized):
        raise ValueError("Use a 1–64 character label with letters, numbers, '_' or '-'.")
    return normalized


def append_sample(
    csv_path: Path,
    label: str,
    capture_session_id: str,
    hand_count: int,
    feature_values: Mapping[str, float],
) -> None:
    """Append one labeled sample, refusing incompatible existing CSV schemas."""
    label = validate_label(label)
    missing = set(FEATURE_COLUMNS).difference(feature_values)
    if missing:
        raise ValueError(f"Feature row is missing {len(missing)} required values.")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    if file_exists:
        with csv_path.open("r", newline="", encoding="utf-8") as existing_file:
            header = next(csv.reader(existing_file), [])
        if header != CSV_COLUMNS:
            raise ValueError(
                f"{csv_path} has an incompatible header. Move it aside before collecting "
                "with this feature format."
            )

    row = {
        "sample_id": str(uuid.uuid4()),
        "capture_session_id": capture_session_id,
        "gesture_label": label,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hand_count": int(hand_count),
        **{name: float(feature_values[name]) for name in FEATURE_COLUMNS},
    }
    with csv_path.open("a", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
