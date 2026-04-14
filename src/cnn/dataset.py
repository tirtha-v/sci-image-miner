"""PyTorch Dataset for panel classification with augmentations."""

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image


class PanelDataset(Dataset):
    """Dataset that loads pre-cropped panel images and labels."""

    def __init__(self, csv_path: str, label2idx: dict[str, int], transform=None):
        self.samples = []
        self.label2idx = label2idx
        self.transform = transform

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row["label"].strip().lower()
                if label in label2idx:
                    self.samples.append((row["image_path"], label2idx[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_idx = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label_idx


def build_label_mapping(csv_path: str) -> tuple[dict[str, int], list[str]]:
    """Build label-to-index mapping from CSV.

    Returns:
        (label2idx, idx2label) — sorted alphabetically.
    """
    labels = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.add(row["label"].strip().lower())

    idx2label = sorted(labels)
    label2idx = {l: i for i, l in enumerate(idx2label)}
    return label2idx, idx2label


def get_train_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.RandomGrayscale(p=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def make_weighted_sampler(dataset: PanelDataset) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler for class-imbalanced training."""
    label_counts = Counter(label_idx for _, label_idx in dataset.samples)
    total = len(dataset.samples)
    class_weights = {cls: total / count for cls, count in label_counts.items()}
    sample_weights = [class_weights[label_idx] for _, label_idx in dataset.samples]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_class_weights(dataset: PanelDataset, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss."""
    counts = Counter(label_idx for _, label_idx in dataset.samples)
    total = len(dataset.samples)
    weights = torch.zeros(num_classes)
    for cls_idx in range(num_classes):
        count = counts.get(cls_idx, 1)
        weights[cls_idx] = total / (num_classes * count)
    return weights
