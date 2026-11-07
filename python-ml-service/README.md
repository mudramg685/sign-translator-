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

## Train and evaluate the initial model

Collect each gesture in at least five separate runs of `collect_dataset.py`, with multiple saved examples per run. The training script uses five-fold stratified group splitting: all rows from a capture session stay together, with three folds for training and one fold each for validation and testing.

```powershell
python scripts/train_model.py
```

Options include `--epochs`, `--batch-size`, `--patience`, `--seed`, `--dataset`, `--output-dir`, and `--device cpu|cuda|auto`. CPU is the default fallback. The script trains a small PyTorch feed-forward classifier with early stopping on validation loss and class-weighted loss for imbalanced labels.

It writes `models/gesture_classifier.pt`, `models/evaluation.json`, and `models/confusion_matrix.csv`. The evaluation includes held-out accuracy, macro precision, macro recall, macro and weighted F1, per-class scores, and the confusion matrix. These local model artifacts are excluded from Git. If a session-based split cannot place every label in all three partitions, collect more independent sessions per label.

The test score measures held-out capture sessions from this dataset. The collector does not record signer identity, so the score does not establish generalization to unseen signers. Results also depend on the selected vocabulary, camera conditions, and data collection quality; they do not measure full ISL translation.
