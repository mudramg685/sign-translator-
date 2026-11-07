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

Alternatively, the [Indian sign Language-Real-life Words dataset](https://data.mendeley.com/datasets/s6kgb6r3ss/2) provides 18,000 static images across 20 word labels under CC BY 4.0. Download and extract it from Mendeley Data, then import images whose names follow `word_imageIndex_userN_sampleN.jpg`:

```powershell
python scripts/import_mendeley_images.py --input path/to/extracted/dataset
```

The importer extracts the same normalized hand landmarks and groups all samples by signer so a signer cannot appear in both training and held-out partitions. It writes `dataset/processed/mendeley_landmarks.csv`. Use `--overwrite` to replace a prior generated CSV. The source description reports eight participants but specifies participant IDs 1–6 in the filename convention; check the extracted files and the importer's signer counts before training. The grouped training script requires at least five signers/sessions per label.

Dataset attribution: Tyagi, A. and Bansal, S. *Indian sign Language-Real-life Words*, Mendeley Data, version 2, 2022, DOI [10.17632/s6kgb6r3ss.2](https://doi.org/10.17632/s6kgb6r3ss.2), CC BY 4.0. Include this citation and the [Creative Commons Attribution 4.0 terms](https://creativecommons.org/licenses/by/4.0/) when redistributing derived work. Do not commit the source images or extracted landmarks to this repository.

For webcam-collected rows in `dataset/raw/landmarks.csv`, run:

```powershell
python scripts/train_model.py
```

For the imported Mendeley rows, set the dataset path explicitly:

```powershell
python scripts/train_model.py --dataset dataset/processed/mendeley_landmarks.csv
```

Options include `--epochs`, `--batch-size`, `--patience`, `--seed`, `--dataset`, `--output-dir`, and `--device cpu|cuda|auto`. CPU is the default fallback. The script trains a small PyTorch feed-forward classifier with early stopping on validation loss and class-weighted loss for imbalanced labels.

It writes `models/gesture_classifier.pt`, `models/evaluation.json`, and `models/confusion_matrix.csv`. The evaluation includes held-out accuracy, macro precision, macro recall, macro and weighted F1, per-class scores, and the confusion matrix. These local model artifacts are excluded from Git. If a session-based split cannot place every label in all three partitions, collect more independent sessions per label.

The test score measures held-out capture sessions from this dataset. The collector does not record signer identity, so the score does not establish generalization to unseen signers. Results also depend on the selected vocabulary, camera conditions, and data collection quality; they do not measure full ISL translation.
