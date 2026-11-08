"""Evaluate a saved WordSignModel checkpoint and save word_vocab.json + word_eval_metrics.json.

Run after training if the main script was interrupted before saving artifacts.
Usage:
    python backend/scripts/eval_word_model.py
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.recognition.word_model import WordSignModel  # noqa: E402
from backend.scripts.word_dataset import load_word_dataset  # noqa: E402

DEFAULT_DATASET = ROOT / "backend" / "data" / "processed" / "word_dataset.npz"
DEFAULT_META = ROOT / "backend" / "data" / "processed" / "meta.json"
DEFAULT_MODELS_DIR = ROOT / "backend" / "models"


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            total_loss += loss.item() * batch_x.size(0)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(batch_y.cpu().tolist())
            all_probs.append(probs.cpu().numpy())
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc, np.array(all_preds), np.array(all_targets), np.concatenate(all_probs, axis=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = DEFAULT_MODELS_DIR
    checkpoint_path = models_dir / "word_model.pt"

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    vocabulary = checkpoint["vocabulary"]
    num_classes = checkpoint["num_classes"]
    total_params = checkpoint["total_params"]
    best_epoch = checkpoint["epoch"]
    best_val_acc = checkpoint["val_acc"]
    best_val_loss = checkpoint["val_loss"]

    print(f"  Checkpoint epoch: {best_epoch}")
    print(f"  Best val accuracy: {best_val_acc:.4f}")
    print(f"  Vocabulary: {num_classes} classes")
    print(f"  Parameters: {total_params:,}")

    dataset = load_word_dataset(DEFAULT_DATASET, DEFAULT_META)
    if vocabulary != dataset.vocabulary:
        raise ValueError("Checkpoint vocabulary does not match the approved dataset mapping.")
    expected_label_to_id = {word: index for index, word in enumerate(dataset.vocabulary)}
    if "label_to_id" in checkpoint and checkpoint["label_to_id"] != expected_label_to_id:
        raise ValueError("Checkpoint label_to_id does not match the approved vocabulary mapping.")

    X_test, y_test = dataset.splits["test"]
    print(f"  Test samples: {len(y_test)} (held-out test split)")

    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Rebuild model
    model = WordSignModel(
        in_features=104,
        num_classes=num_classes,
        cnn_channels=(128, 128),
        gru_hidden_size=128,
        num_gru_layers=2,
        dropout=0.3,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model weights loaded successfully.")
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameter_count != total_params:
        raise ValueError(
            f"Checkpoint parameter count {total_params:,} does not match model "
            f"parameter count {parameter_count:,}."
        )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    X_train, y_train = dataset.splits["train"]
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    train_loss, train_acc, _, _, _ = evaluate(model, train_loader, criterion, device)
    print(
        f"Train split loss at selected checkpoint: {train_loss:.4f} "
        f"(accuracy {train_acc:.4f}; evaluation mode)"
    )

    # Evaluate the held-out split only after loading the selected checkpoint.
    test_loss, test_acc, test_preds, test_targets, _ = evaluate(
        model, test_loader, criterion, device
    )
    print(f"\nHeld-Out Test Accuracy: {test_acc:.4f} (900 samples across 4 unseen test signers)")

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_targets, test_preds, average="macro", zero_division=0
    )
    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
        test_targets, test_preds, average=None, zero_division=0
    )
    cm = confusion_matrix(test_targets, test_preds, labels=list(range(num_classes)))

    accuracy_gate = 0.75
    passed_gate = test_acc >= accuracy_gate

    per_word_results = {}
    failing_words = []

    print("\n" + "=" * 80)
    print(f"{'Sign':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Test Acc':<10} | Status")
    print("=" * 80)
    for idx, word in enumerate(vocabulary):
        class_total = int((test_targets == idx).sum())
        class_correct = int(((test_targets == idx) & (test_preds == idx)).sum())
        class_acc = class_correct / max(class_total, 1)
        p = float(per_class_p[idx])
        r = float(per_class_r[idx])
        f = float(per_class_f1[idx])
        status = "PASS" if class_acc >= 0.60 else "LOW"
        if class_acc < 0.60:
            failing_words.append(word)
        per_word_results[word] = {"precision": p, "recall": r, "f1_score": f, "accuracy": class_acc, "support": class_total}
        print(f"{word:<16} | {p:<10.4f} | {r:<10.4f} | {f:<10.4f} | {class_acc:<10.4f} | {status}")

    print("=" * 80)
    print("\nOverall Test Summary:")
    print(f"  Test Accuracy:    {test_acc:.4f} ({'PASSED GATE >= 0.75' if passed_gate else 'BELOW GATE'})")
    print(f"  Macro Precision:  {precision:.4f}")
    print(f"  Macro Recall:     {recall:.4f}")
    print(f"  Macro F1:         {f1:.4f}")
    print(f"  Classes < 0.60:   {len(failing_words)} ({', '.join(failing_words) if failing_words else 'None'})")

    # Save word_vocab.json
    vocab_file = models_dir / "word_vocab.json"
    with open(vocab_file, "w", encoding="utf-8") as fout:
        json.dump(
            {
                "vocabulary": vocabulary,
                "label_to_id": {w: i for i, w in enumerate(vocabulary)},
                "id_to_label": {str(i): w for i, w in enumerate(vocabulary)},
            },
            fout,
            indent=2,
        )
    print(f"\nSaved: {vocab_file}")

    # Keep training-only measurements when this script is run after training.
    metrics_file = models_dir / "word_eval_metrics.json"
    previous_metrics = {}
    if metrics_file.exists():
        with metrics_file.open(encoding="utf-8") as file:
            previous_metrics = json.load(file)

    metrics_summary = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "model_architecture": "WordSignModel(1D CNN + BiGRU)",
        "trainable_parameters": total_params,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_acc,
        "train_split_loss_at_checkpoint": train_loss,
        "train_split_accuracy_at_checkpoint": train_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "accuracy_gate_threshold": accuracy_gate,
        "passed_accuracy_gate": passed_gate,
        "failing_words": failing_words,
        "per_word_results": per_word_results,
        "confusion_matrix": cm.tolist(),
    }
    for training_key in (
        "training_duration_seconds",
        "final_train_loss",
        "best_epoch_train_loss",
        "training_history",
    ):
        if training_key in previous_metrics:
            metrics_summary[training_key] = previous_metrics[training_key]
    with open(metrics_file, "w", encoding="utf-8") as fout:
        json.dump(metrics_summary, fout, indent=2)
    print(f"Saved: {metrics_file}")
    print(f"\nDone. Gate {'PASSED' if passed_gate else 'FAILED'}: Test Acc={test_acc:.4f} (threshold=0.75)")


if __name__ == "__main__":
    main()
