"""Print basic label counts for a collected landmark CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))

from app.dataset import CSV_COLUMNS

DEFAULT_DATASET = SERVICE_ROOT / "dataset" / "raw" / "landmarks.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET, help="Dataset CSV path")
    args = parser.parse_args()
    path = args.input.expanduser().resolve()
    if not path.is_file():
        print(f"Dataset not found: {path}", file=sys.stderr)
        return 2

    counts: Counter[str] = Counter()
    total = 0
    try:
        with path.open("r", newline="", encoding="utf-8") as dataset_file:
            reader = csv.DictReader(dataset_file)
            if reader.fieldnames != CSV_COLUMNS:
                raise ValueError("CSV header does not match the current landmark feature format.")
            for row in reader:
                counts[row["gesture_label"]] += 1
                total += 1
    except (OSError, csv.Error, ValueError) as error:
        print(f"Could not inspect dataset: {error}", file=sys.stderr)
        return 3

    print(f"Dataset: {path}")
    print(f"Total samples: {total}")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")
    if len(counts) < 2:
        print("Collect at least two different labels before training a classifier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
