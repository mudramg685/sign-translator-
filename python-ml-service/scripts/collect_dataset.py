"""Collect labeled hand-landmark samples from a webcam.

Run from python-ml-service:
    python scripts/collect_dataset.py --label hello

Press SPACE to save the currently detected hand(s). Press Q or ESC to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))

import cv2
import mediapipe as mp

from app.dataset import append_sample, validate_label
from app.landmarks import frame_feature_values

DEFAULT_OUTPUT = SERVICE_ROOT / "dataset" / "raw" / "landmarks.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Gesture class label, e.g. hello")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument(
        "--capture-interval",
        type=float,
        default=0.35,
        help="Minimum seconds between saved samples (default: 0.35)",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.65,
        help="MediaPipe hand detection threshold from 0 to 1 (default: 0.65)",
    )
    args = parser.parse_args()
    try:
        args.label = validate_label(args.label)
    except ValueError as error:
        parser.error(str(error))
    if args.capture_interval < 0:
        parser.error("--capture-interval cannot be negative")
    if not 0.0 <= args.min_detection_confidence <= 1.0:
        parser.error("--min-detection-confidence must be between 0 and 1")
    return args


def main() -> int:
    args = parse_args()
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(
            f"Could not open camera {args.camera}. Check camera permissions or choose another "
            "index with --camera.",
            file=sys.stderr,
        )
        return 2

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    saved_count = 0
    capture_session_id = str(uuid.uuid4())
    last_capture_at = 0.0
    output_path = args.output.expanduser().resolve()
    print(f"Collecting label '{args.label}' to {output_path}")
    print("Show a clear sign; press SPACE to save a sample. Press Q or ESC to exit.")

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=0.55,
        ) as hands:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Camera stopped returning frames.", file=sys.stderr)
                    return 3

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = hands.process(rgb_frame)
                rgb_frame.flags.writeable = True

                landmarks = results.multi_hand_landmarks or []
                handedness = results.multi_handedness or []
                features = frame_feature_values(landmarks, handedness)
                for hand_landmarks in landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                status = "Hand detected - press SPACE to save" if features else "Show one or two hands"
                color = (80, 220, 120) if features else (60, 180, 255)
                cv2.putText(frame, f"Label: {args.label}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, status, (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2, cv2.LINE_AA)
                cv2.putText(frame, f"Saved: {saved_count}   [SPACE] capture  [Q] quit",
                            (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("ISL landmark dataset collector", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == 32:
                    now = time.monotonic()
                    if now - last_capture_at < args.capture_interval:
                        continue
                    if features is None:
                        print("No hand landmarks found; sample skipped.")
                        continue
                    append_sample(
                        output_path,
                        args.label,
                        capture_session_id,
                        len(landmarks),
                        features,
                    )
                    saved_count += 1
                    last_capture_at = now
                    print(f"Saved sample {saved_count} for '{args.label}'.")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Collection stopped: {error}", file=sys.stderr)
        return 4
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Finished. Saved {saved_count} samples to {output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
