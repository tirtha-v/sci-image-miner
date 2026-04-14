"""CNN training loop with 2-stage transfer learning."""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from .dataset import (
    PanelDataset,
    build_label_mapping,
    get_train_transform,
    get_val_transform,
    make_weighted_sampler,
    get_class_weights,
)
from .model import create_model, freeze_backbone, unfreeze_all, MODEL_CONFIGS


class FocalLoss(nn.Module):
    """Focal loss for imbalanced multi-class classification.

    Down-weights easy/majority examples so training focuses on hard/minority classes.
    """

    def __init__(self, weight=None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    total = len(all_labels)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return {
        "loss": total_loss / total,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def train(
    model_name: str = "efficientnet_b0",
    train_csv: str = "data/train_panels.csv",
    dev_csv: str = "data/dev_panels.csv",
    output_dir: str = "outputs/cnn",
    epochs: int = 30,
    frozen_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    patience: int = 7,
    device: str = "cuda:0",
    use_focal: bool = False,
):
    """Run 2-stage transfer learning training."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = MODEL_CONFIGS[model_name]
    img_size = config["img_size"]
    timm_name = config["timm_name"]
    timm_img_size = config.get("timm_img_size", None)  # only passed to timm for attention models

    # Build label mapping from training data
    label2idx, idx2label = build_label_mapping(train_csv)
    num_classes = len(label2idx)
    print(f"Classes: {num_classes}")
    print(f"Model: {timm_name} (img_size={img_size})")

    # Save label mapping
    with open(output_dir / "label_mapping.json", "w") as f:
        json.dump({"label2idx": label2idx, "idx2label": idx2label}, f, indent=2)

    # Datasets
    train_dataset = PanelDataset(train_csv, label2idx, get_train_transform(img_size))
    dev_dataset = PanelDataset(dev_csv, label2idx, get_val_transform(img_size))
    print(f"Train: {len(train_dataset)}, Dev: {len(dev_dataset)}")

    sampler = make_weighted_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Model
    model = create_model(timm_name, num_classes, pretrained=True, img_size=timm_img_size)
    model = model.to(device)

    # Loss with class weights. NOTE: label_smoothing is intentionally NOT combined
    # with class_weights — the two interact badly: smoothing redistributes probability
    # to rare (high-weight) classes, producing exploding gradients toward rare labels.
    # Label smoothing is applied separately in Stage 2 via criterion_smooth (no weights).
    # Focal loss (use_focal=True) is an alternative that handles imbalance without
    # needing separate class weights — gamma=2 down-weights easy majority examples.
    class_weights = get_class_weights(train_dataset, num_classes).to(device)
    if use_focal:
        print("[train] Using Focal Loss (gamma=2.0) for both stages")
        criterion = FocalLoss(weight=class_weights, gamma=2.0)
        criterion_smooth = FocalLoss(gamma=2.0, label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        criterion_smooth = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Stage 1: Frozen backbone
    print(f"\n--- Stage 1: Frozen backbone ({frozen_epochs} epochs) ---")
    freeze_backbone(model)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr * 10)
    scheduler = CosineAnnealingLR(optimizer, T_max=frozen_epochs)

    best_macro_f1 = 0
    patience_counter = 0
    history = []

    for epoch in range(1, frozen_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_metrics = evaluate(model, dev_loader, criterion, device, num_classes)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{frozen_epochs} ({elapsed:.0f}s) | "
            f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
            f"macro_f1={val_metrics['macro_f1']:.4f}"
        )
        history.append({
            "epoch": epoch, "stage": 1,
            "train_loss": train_loss, "train_acc": train_acc,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1

    # Stage 2: Full finetuning with layerwise LR (backbone at 0.1x to preserve ImageNet features)
    print(f"\n--- Stage 2: Full finetuning ({epochs - frozen_epochs} epochs) ---")
    unfreeze_all(model)
    _head_keywords = ("classifier", "classif", "fc", "head")
    backbone_params = [p for n, p in model.named_parameters() if not any(kw in n for kw in _head_keywords)]
    head_params = [p for n, p in model.named_parameters() if any(kw in n for kw in _head_keywords)]
    optimizer = AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": head_params, "lr": lr},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - frozen_epochs)

    for epoch in range(frozen_epochs + 1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion_smooth, optimizer, device)
        scheduler.step()
        val_metrics = evaluate(model, dev_loader, criterion_smooth, device, num_classes)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{epochs} ({elapsed:.0f}s) | "
            f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
            f"macro_f1={val_metrics['macro_f1']:.4f}"
        )
        history.append({
            "epoch": epoch, "stage": 2,
            "train_loss": train_loss, "train_acc": train_acc,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    # Save history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest dev macro F1: {best_macro_f1:.4f}")
    print(f"Model saved to {output_dir / 'best_model.pt'}")
    return best_macro_f1
