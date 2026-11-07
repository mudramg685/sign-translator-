"""Convert the CC BY 4.0 Mendeley ISL word images into landmark CSV rows.

Download and extract the dataset yourself, then run from python-ml-service:
    python scripts/import_mendeley_images.py --input path/to/extracted/images

Expected filename format: word_imageIndex_userN_sampleN.jpg
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))

import cv2
import mediapipe as mp

from app.dataset import append_sample
from app.landmarks import frame_feature_values

DEFAULT_OUTPUT = SERVICE_ROOT / "dataset" / "processed" / "mendeley_landmarks.csv"
IMAGE_NAME = re.compile(
    r"^(?P<label>[a-z]+)_(?P<image_index>\d+)_(?P<signer>user\d+)_(?P<sample>\d+)\.(?:jpg|jpeg|png)$",
    re.IGNORECASE,
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Extracted dataset directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Landmark CSV output")
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.55,
        help="MediaPipe detection threshold from 0 to 1",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output CSV after a successful import",
    )
    args = parser.parse_args()
    if not 0 <= args.min_detection_confidence <= 1:
        parser.error("--min-detection-confidence must be between 0 and 1")
    return args


def main() -> int:
    args = parse_args()
    source_dir = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not source_dir.is_dir():
        print(f"Input directory not found: {source_dir}", file=sys.stderr)
        return 2
    if output_path.exists() and not args.overwrite:
        print(
            f"Output already exists: {output_path}. Choose another --output or pass --overwrite.",
            file=sys.stderr,
        )
        return 2

    image_files = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_files:
        print(f"No JPG or PNG images found under {source_dir}.", file=sys.stderr)
        return 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = tempfile.NamedTemporaryFile(
        prefix="isl-landmarks-", suffix=".csv", dir=output_path.parent, delete=False
    )
    temporary_path = Path(temporary_file.name)
    temporary_file.close()

    counts: Counter[str] = Counter()
    successful_signers: dict[str, set[str]] = defaultdict(set)
    unmatched = 0
    try:
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=args.min_detection_confidence,
        ) as hands:
            for image_number, image_path in enumerate(image_files, start=1):
                match = IMAGE_NAME.fullmatch(image_path.name)
                if not match:
                    unmatched += 1
                    continue

                image = cv2.imread(str(image_path))
                if image is None:
                    counts["unreadable"] += 1
                    continue
                image = cv2.flip(image, 1)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb_image)
                landmarks = result.multi_hand_landmarks or []
                handedness = result.multi_handedness or []
                features = frame_feature_values(landmarks, handedness)
                if features is None:
                    counts["no_hand_detected"] += 1
                    continue

                label = match.group("label").lower()
                signer = match.group("signer").lower()
                append_sample(
                    temporary_path,
                    label,
                    f"mendeley_{signer}",
                    len(landmarks),
                    features,
                )
                successful_signers[label].add(signer)
                counts[label] += 1
                if image_number % 500 == 0:
                    print(f"Processed {image_number}/{len(image_files)} images...")

        if sum(counts[label] for label in successful_signers) == 0:
            print("No usable images matched the expected naming scheme; no output was written.",
                  file=sys.stderr)
            return 3
        temporary_path.replace(output_path)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Import stopped: {error}. Partial output was discarded.", file=sys.stderr)
        return 3
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    imported = sum(counts[label] for label in successful_signers)
    print(f"Imported {imported} usable images from {len(image_files)} image files.")
    print(f"Output: {output_path}")
    if unmatched:
        print(f"Skipped {unmatched} filenames that did not match the documented pattern.")
    if counts["unreadable"] or counts["no_hand_detected"]:
        print(
            f"Skipped unreadable={counts['unreadable']} and no-hand="
            f"{counts['no_hand_detected']} images."
        )
    for label in sorted(successful_signers):
        signer_count = len(successful_signers[label])
        print(f"  {label}: {counts[label]} images from {signer_count} signers")
        if signer_count < 5:
            print(
                f"    Warning: {label} has fewer than five signers; the grouped "
                "train/validation/test split will likely need more data."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
