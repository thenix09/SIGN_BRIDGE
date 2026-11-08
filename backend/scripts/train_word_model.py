"""Training and evaluation script for SignBridge 30-word isolated sign recognition."""

from __future__ import annotations

import argparse
import datetime
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

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


def set_seed(seed: int = 42) -> None:
    """Set random seed across python, numpy, and torch for determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model on a DataLoader.

    Returns:
        (avg_loss, accuracy, all_preds, all_targets, all_probs)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds: list[int] = []
    all_targets: list[int] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

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


def train_word_model(
    dataset_path: Path = DEFAULT_DATASET,
    meta_path: Path = DEFAULT_META,
    models_dir: Path = DEFAULT_MODELS_DIR,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    patience: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Train word recognition model with early stopping on validation accuracy."""
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, and patience must all be positive.")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = load_word_dataset(dataset_path, meta_path)
    vocabulary = dataset.vocabulary
    num_classes = len(vocabulary)
    tensors = dataset.splits
    print(f"Validated dataset: {dataset_path}")
    print(f"  Train samples: {len(tensors['train'][1])}")
    print(f"  Val samples:   {len(tensors['val'][1])}")
    print(f"  Test samples:  {len(tensors['test'][1])} (reserved until checkpoint selection)")

    X_train, y_train = tensors["train"]
    X_val, y_val = tensors["val"]

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = WordSignModel(
        in_features=104,
        num_classes=num_classes,
        cnn_channels=(128, 128),
        gru_hidden_size=128,
        num_gru_layers=2,
        dropout=0.3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"WordSignModel initialized: {total_params:,} trainable parameters (< 2M)")
    if total_params >= 2_000_000:
        raise ValueError(f"Trainable parameter limit exceeded: {total_params:,} >= 2,000,000.")

    # Optimizer, Loss, Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    best_checkpoint_path = models_dir / "word_model.pt"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print(f"{'Epoch':^7} | {'Train Loss':^11} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10} | {'Status':^12}")
    print("=" * 75)

    start_train_time = time.time()
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)

        scheduler.step()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation evaluation
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)

        status = ""
        # Checkpoint on validation accuracy improvement
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            status = "* Best (Saved)"

            # Save checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "vocabulary": vocabulary,
                    "label_to_id": {word: index for index, word in enumerate(vocabulary)},
                    "id_to_label": {str(index): word for index, word in enumerate(vocabulary)},
                    "num_classes": num_classes,
                    "total_params": total_params,
                    "seed": seed,
                    "model_config": {
                        "in_features": 104,
                        "cnn_channels": [128, 128],
                        "gru_hidden_size": 128,
                        "num_gru_layers": 2,
                        "dropout": 0.3,
                    },
                },
                best_checkpoint_path,
            )
        else:
            patience_counter += 1
            status = f"patience {patience_counter}/{patience}"

        history.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        print(
            f"{epoch:^7d} | {epoch_train_loss:^11.4f} | {epoch_train_acc:^10.4f} | "
            f"{val_loss:^10.4f} | {val_acc:^10.4f} | {status:^12}"
        )

        if patience_counter >= patience:
            print(f"\n[Early Stopping] Triggered after {epoch} epochs (Best Epoch: {best_epoch}, Best Val Acc: {best_val_acc:.4f})")
            break

    total_training_time = time.time() - start_train_time
    print("=" * 75)
    print(f"Training completed in {total_training_time:.2f} seconds ({total_training_time / 60:.1f} minutes)")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    # -------------------------------------------------------------
    # Final Held-Out Test Evaluation (Loaded strictly from best checkpoint)
    # -------------------------------------------------------------
    print("\nLoading best model checkpoint for final held-out test evaluation...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    X_test, y_test = tensors["test"]
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loss, test_acc, test_preds, test_targets, _ = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nHeld-Out Test Accuracy: {test_acc:.4f} (900 samples across 4 unseen test signers)")

    # Compute Macro Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_targets, test_preds, average="macro", zero_division=0
    )

    # Compute Per-Class Metrics
    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
        test_targets, test_preds, average=None, zero_division=0
    )

    # Confusion Matrix
    cm = confusion_matrix(test_targets, test_preds, labels=list(range(num_classes)))

    # Gate Assessment
    accuracy_gate = 0.75  # SIGNBRIDGE_WORD_GATE requirement from MODEL_CARD
    passed_gate = test_acc >= accuracy_gate

    per_word_results: dict[str, dict[str, float]] = {}
    failing_words: list[str] = []

    print("\n" + "=" * 75)
    print(f"{'Class (Sign)':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Test Acc':<10} | {'Status':<8}")
    print("=" * 75)
    for idx, word in enumerate(vocabulary):
        # Accuracy for class idx (correct / total for that class in test)
        class_total = int((test_targets == idx).sum())
        class_correct = int(((test_targets == idx) & (test_preds == idx)).sum())
        class_acc = class_correct / max(class_total, 1)

        p = float(per_class_p[idx])
        r = float(per_class_r[idx])
        f = float(per_class_f1[idx])

        status = "PASS" if class_acc >= 0.60 else "LOW"
        if class_acc < 0.60:
            failing_words.append(word)

        per_word_results[word] = {
            "precision": p,
            "recall": r,
            "f1_score": f,
            "accuracy": class_acc,
            "support": class_total,
        }

        print(f"{word:<16} | {p:<10.4f} | {r:<10.4f} | {f:<10.4f} | {class_acc:<10.4f} | {status:<8}")

    print("=" * 75)
    print("\nOverall Test Summary:")
    print(f"  Test Accuracy:    {test_acc:.4f} ({'PASSED GATE >= 0.75' if passed_gate else 'BELOW GATE'})")
    print(f"  Macro Precision:  {precision:.4f}")
    print(f"  Macro Recall:     {recall:.4f}")
    print(f"  Macro F1:         {f1:.4f}")
    print(f"  Classes < 0.60:   {len(failing_words)} ({', '.join(failing_words) if failing_words else 'None'})")

    # Save artifacts in backend/models
    # 1. word_vocab.json
    vocab_file = models_dir / "word_vocab.json"
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "vocabulary": vocabulary,
                "label_to_id": {w: i for i, w in enumerate(vocabulary)},
                "id_to_label": {str(i): w for i, w in enumerate(vocabulary)},
            },
            f,
            indent=2,
        )
    print(f"\nSaved vocabulary mapping to: {vocab_file}")

    # 2. word_eval_metrics.json
    metrics_file = models_dir / "word_eval_metrics.json"
    metrics_summary = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "model_architecture": "WordSignModel(1D CNN + BiGRU)",
        "trainable_parameters": total_params,
        "training_duration_seconds": total_training_time,
        "final_train_loss": history[-1]["train_loss"],
        "best_epoch_train_loss": history[best_epoch - 1]["train_loss"],
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "accuracy_gate_threshold": accuracy_gate,
        "passed_accuracy_gate": passed_gate,
        "failing_words": failing_words,
        "per_word_results": per_word_results,
        "confusion_matrix": cm.tolist(),
        "training_history": history,
    }
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved evaluation metrics to: {metrics_file}")
    print(f"Saved model weights to: {best_checkpoint_path} ({best_checkpoint_path.stat().st_size / 1e6:.2f} MB)")

    return metrics_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train WordSignModel for SignBridge.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_word_model(
        dataset_path=args.dataset,
        meta_path=args.meta,
        models_dir=args.models_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
