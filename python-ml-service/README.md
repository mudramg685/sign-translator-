# Python landmark collection

This module collects labeled hand landmarks from a webcam. It uses MediaPipe Hands, which reports 21 ordered landmarks per hand and left/right handedness. See the [official MediaPipe Hands documentation](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md). It does not guess gesture names or produce predictions; a trained classifier is added in a later phase.

## Environment

Use Python 3.11 and create a virtual environment from this directory:

```powershell
cd python-ml-service
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`. If you only want to collect the dataset at this phase, you can install `mediapipe`, `opencv-python`, and `numpy`; the full requirements also include the future API and training dependencies.

## Collect samples

From `python-ml-service`, collect examples for one label at a time:

```powershell
python scripts/collect_dataset.py --label hello
python scripts/collect_dataset.py --label yes
```

Use a simple label made of letters, numbers, underscores, or hyphens. The webcam window is mirrored for a natural selfie view. Hold the sign clearly and press **Space** to save a sample. Press **Q** or **Esc** to stop. Change the camera index with `--camera 1`, or choose a different CSV with `--output path/to/file.csv`.

The collector accepts one or two detected hands and skips frames without usable landmarks. It saves a row only when you press Space, with a short capture interval to avoid accidental rapid duplicates. Vary distance, position, lighting, and signer between examples while keeping each label's intended meaning consistent.

## CSV format

`dataset/raw/landmarks.csv` stores a sample UUID, a capture-session UUID, gesture label, UTC capture time, detected hand count, then 128 model features: presence plus 21 normalized `(x, y, z)` points for each handedness slot. Each collector run has its own session ID so later train/validation/test splits can keep frames from one session together. Each hand is translated to its wrist and scaled by its farthest landmark; orientation is preserved. Missing hand slots are zeros. The column ordering is defined in `app/landmarks.py` and is shared with future training and inference code.

Raw participant samples are ignored by Git. Keep a separate backup if you need to preserve a collected dataset. Do not collect or share samples without the participant's permission.

Inspect class balance with:

```powershell
python scripts/inspect_dataset.py
```

The data is not ready for model training merely because it has rows. Phase 4 will add stratified train/validation/test splitting and evaluation. Capture examples from multiple sessions and, where possible, multiple consenting signers to reduce memorization.
