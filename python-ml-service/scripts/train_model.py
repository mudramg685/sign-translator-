"""Train and evaluate a grouped-split PyTorch landmark classifier.

Run from python-ml-service after collecting the dataset:
    python scripts/train_model.py
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.dataset import CSV_COLUMNS
from app.landmarks import FEATURE_COLUMNS
from app.model import LandmarkClassifier

DEFAULT_DATASET = SERVICE_ROOT / "dataset" / "raw" / "landmarks.csv"
DEFAULT_OUTPUT_DIR = SERVICE_ROOT / "models"
MIN_SESSIONS_PER_CLASS = 5
HIDDEN_SIZES = (128, 64)
DROPOUT = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch-size, and patience must be positive")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but PyTorch cannot access a CUDA device")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_dataset(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    if not path.is_file():
        raise ValueError(f"Dataset not found: {path}. Collect labeled samples first.")
    frame = pd.read_csv(path)
    if list(frame.columns) != CSV_COLUMNS:
        raise ValueError(
            "Dataset columns do not match the current collector format. "
            "Use the CSV written by scripts/collect_dataset.py."
        )
    if frame.empty:
        raise ValueError("Dataset contains only a header; collect samples before training.")
    if frame["sample_id"].duplicated().any():
        raise ValueError("Dataset contains duplicate sample_id values.")
    if frame["capture_session_id"].isna().any() or frame["gesture_label"].isna().any():
        raise ValueError("Each sample must have a gesture label and capture session ID.")

    features_frame = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    features = features_frame.to_numpy(dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Landmark features contain missing or non-finite numbers.")

    labels = frame["gesture_label"].astype(str).to_numpy()
    groups = frame["capture_session_id"].astype(str).to_numpy()
    classes = sorted(set(labels))
    if len(classes) < 2:
        raise ValueError("Training needs at least two gesture labels.")

    sessions_per_class = frame.groupby("gesture_label")["capture_session_id"].nunique()
    insufficient = sessions_per_class[sessions_per_class < MIN_SESSIONS_PER_CLASS]
    if not insufficient.empty:
        details = ", ".join(f"{label}: {count}" for label, count in insufficient.items())
        raise ValueError(
            f"Each label needs at least {MIN_SESSIONS_PER_CLASS} separate capture sessions "
            f"for a five-fold session split. Current counts: {details}. Collect each label "
            "in several fresh runs of collect_dataset.py."
        )
    return frame, features, labels, classes


def make_grouped_split(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    classes: list[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reserve one fifth each for validation and test, grouped by capture session."""
    all_classes = set(classes)
    for attempt in range(100):
        splitter = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed + attempt,
        )
        folds = [test_indices for _, test_indices in splitter.split(features, labels, groups)]
        for test_fold in range(len(folds)):
            for validation_fold in range(len(folds)):
                if validation_fold == test_fold:
                    continue
                train_folds = [
                    folds[index]
                    for index in range(len(folds))
                    if index not in (test_fold, validation_fold)
                ]
                train_indices = np.concatenate(train_folds)
                validation_indices = folds[validation_fold]
                test_indices = folds[test_fold]
                if all(
                    set(labels[indices]) == all_classes
                    for indices in (train_indices, validation_indices, test_indices)
                ):
                    if set(groups[train_indices]).intersection(groups[validation_indices]):
                        continue
                    if set(groups[train_indices]).intersection(groups[test_indices]):
                        continue
                    if set(groups[validation_indices]).intersection(groups[test_indices]):
                        continue
                    return train_indices, validation_indices, test_indices
    raise ValueError(
        "Could not make train/validation/test splits with every label represented and no "
        "capture-session overlap. Collect more independent sessions for each gesture, then "
        "try again."
    )


def make_loader(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32, copy=False)),
        torch.from_numpy(targets.astype(np.int64, copy=False)),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def train_one_epoch(
    model: LandmarkClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    sample_count = 0
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(features), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
        sample_count += len(targets)
    return total_loss / max(sample_count, 1)


def calculate_loss(
    model: LandmarkClassifier,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    sample_count = 0
    with torch.inference_mode():
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            loss = loss_function(model(features), targets)
            total_loss += loss.item() * len(targets)
            sample_count += len(targets)
    return total_loss / max(sample_count, 1)


def predict(
    model: LandmarkClassifier,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(features.astype(np.float32, copy=False)).to(device))
        return logits.argmax(dim=1).cpu().numpy()


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    dataset_path = args.dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    try:
        frame, features, labels, classes = load_dataset(dataset_path)
        train_indices, validation_indices, test_indices = make_grouped_split(
            features,
            labels,
            frame["capture_session_id"].astype(str).to_numpy(),
            classes,
            args.seed,
        )
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as error:
        print(f"Training stopped: {error}", file=sys.stderr)
        return 2

    label_to_index = {label: index for index, label in enumerate(classes)}
    targets = np.array([label_to_index[label] for label in labels], dtype=np.int64)
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    train_loader = make_loader(
        features[train_indices], targets[train_indices], args.batch_size, True, args.seed
    )
    validation_loader = make_loader(
        features[validation_indices], targets[validation_indices], args.batch_size, False, args.seed
    )
    class_counts = np.bincount(targets[train_indices], minlength=len(classes))
    class_weights = len(train_indices) / (len(classes) * class_counts)
    loss_function = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    model = LandmarkClassifier(len(FEATURE_COLUMNS), len(classes), HIDDEN_SIZES, DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_validation_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    best_epoch = 0

    print(
        f"Samples: {len(frame)} | labels: {len(classes)} | "
        f"train/validation/test: {len(train_indices)}/{len(validation_indices)}/{len(test_indices)} "
        f"| device: {device}"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_function, device)
        validation_loss = calculate_loss(model, validation_loader, loss_function, device)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} validation_loss={validation_loss:.4f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.")
                break

    if best_state is None:
        print("Training did not produce a model checkpoint.", file=sys.stderr)
        return 3
    model.load_state_dict(best_state)
    predictions = predict(model, features[test_indices], device)
    actual = targets[test_indices]
    label_indices = list(range(len(classes)))
    report = classification_report(
        actual,
        predictions,
        labels=label_indices,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(actual, predictions, labels=label_indices)

    metrics = {
        "dataset": str(dataset_path),
        "seed": args.seed,
        "split_method": "StratifiedGroupKFold: 60% train, 20% validation, 20% test by capture session",
        "feature_count": len(FEATURE_COLUMNS),
        "labels": classes,
        "sample_counts": {
            "total": len(frame),
            "train": len(train_indices),
            "validation": len(validation_indices),
            "test": len(test_indices),
        },
        "session_counts": {
            "total": int(frame["capture_session_id"].nunique()),
            "train": int(frame.iloc[train_indices]["capture_session_id"].nunique()),
            "validation": int(frame.iloc[validation_indices]["capture_session_id"].nunique()),
            "test": int(frame.iloc[test_indices]["capture_session_id"].nunique()),
        },
        "session_ids": {
            "train": sorted(frame.iloc[train_indices]["capture_session_id"].astype(str).unique()),
            "validation": sorted(
                frame.iloc[validation_indices]["capture_session_id"].astype(str).unique()
            ),
            "test": sorted(frame.iloc[test_indices]["capture_session_id"].astype(str).unique()),
        },
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "test": {
            "accuracy": accuracy_score(actual, predictions),
            "macro_precision": precision_score(
                actual, predictions, labels=label_indices, average="macro", zero_division=0
            ),
            "macro_recall": recall_score(
                actual, predictions, labels=label_indices, average="macro", zero_division=0
            ),
            "macro_f1": f1_score(
                actual, predictions, labels=label_indices, average="macro", zero_division=0
            ),
            "weighted_f1": f1_score(
                actual, predictions, labels=label_indices, average="weighted", zero_division=0
            ),
            "per_class": report,
            "confusion_matrix": {
                "labels_in_order": classes,
                "rows_actual_columns_predicted": matrix.tolist(),
            },
        },
    }
    checkpoint = {
        "model_state_dict": {name: value.detach().cpu() for name, value in best_state.items()},
        "model_config": {
            "input_size": len(FEATURE_COLUMNS),
            "class_count": len(classes),
            "hidden_sizes": list(HIDDEN_SIZES),
            "dropout": DROPOUT,
        },
        "labels": classes,
        "feature_columns": FEATURE_COLUMNS,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "seed": args.seed,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "gesture_classifier.pt"
    metrics_path = output_dir / "evaluation.json"
    matrix_path = output_dir / "confusion_matrix.csv"
    torch.save(checkpoint, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(matrix, index=classes, columns=classes).rename_axis("actual").to_csv(matrix_path)

    print("\nHeld-out test metrics:")
    print(
        f"accuracy={metrics['test']['accuracy']:.3f} "
        f"macro_precision={metrics['test']['macro_precision']:.3f} "
        f"macro_recall={metrics['test']['macro_recall']:.3f} "
        f"macro_f1={metrics['test']['macro_f1']:.3f} "
        f"weighted_f1={metrics['test']['weighted_f1']:.3f}"
    )
    print(f"Saved model: {model_path}")
    print(f"Saved evaluation: {metrics_path}")
    print(f"Saved confusion matrix: {matrix_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
