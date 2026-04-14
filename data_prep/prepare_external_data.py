#!/usr/bin/env python3
"""Prepare ACL-Fig and DocFigure datasets for training.

Maps external category labels to our competition taxonomy and writes
augmented train CSVs that can be merged with data/train_panels.csv.

Usage:
    python data_prep/prepare_external_data.py --aclfig --output data/train_augmented.csv
    python data_prep/prepare_external_data.py --docfigure data/external/DocFigure_dataset --output data/train_augmented.csv
    python data_prep/prepare_external_data.py --aclfig --docfigure data/external/DocFigure_dataset --output data/train_augmented.csv
"""

import argparse
import os
import shutil
from pathlib import Path
import pandas as pd


# ── ACL-Fig category → competition taxonomy ─────────────────────────────────
# Unmapped (skip): NLP text_grammar_eg, Screenshots, algorithms, graph,
#                  maps, pareto, trees, venn diagram, word cloud
ACLFIG_MAP = {
    "Line graph_chart":   "line chart",
    "bar charts":         "bar chart",
    "boxplots":           "box plot",
    "scatter plot":       "scatter plot",
    "pie chart":          None,               # not in dev set support, skip
    "confusion matrix":   "heatmap",
    "architecture diagram": "conceptual diagram",
    "neural networks":    "conceptual diagram",
    "natural images":     "image panel",
    "tables":             "comparison table",
    # skip the rest
    "NLP text_grammar_eg": None,
    "Screenshots":        None,
    "algorithms":         None,
    "graph":              None,
    "maps":               None,
    "pareto":             None,
    "trees":              None,
    "venn diagram":       None,
    "word cloud":         None,
}

# ── DocFigure category → competition taxonomy ────────────────────────────────
# Actual DocFigure categories from annotation/train.txt (with leading space after comma).
# Format: "filename.png, Category Name" → strip() gives category name.
DOCFIGURE_MAP = {
    "Graph plots":       None,             # too generic/ambiguous
    "Natural images":    "image panel",
    "Tables":            "comparison table",
    "Bar plots":         "bar chart",
    "Scatter plot":      "scatter plot",
    "Heat map":          "heatmap",
    "Flow chart":        "process flow diagram",
    "Block diagram":     "conceptual diagram",
    "Confusion matrix":  "heatmap",
    "Histogram":         None,             # not in our taxonomy (close to bar chart but distinct)
    "Box plot":          "box plot",
    "Contour plot":      None,             # rare, not in our taxonomy
    "Tree Diagram":      "tree diagram",
    "Bubble Chart":      "bubble chart",
    "Area chart":        "area chart",
    "3D objects":        None,             # not in our taxonomy
    "Line Graph":        "line chart",
    "Pie chart":         None,             # rare in our dev set
    "Venn Diagram":      None,
    "Violin plot":       None,
    "Word cloud":        None,
    "Map":               None,
}


def prepare_aclfig(output_dir: Path) -> pd.DataFrame:
    from datasets import load_dataset
    print("[ACL-Fig] Loading dataset...")
    ds = load_dataset("citeseerx/ACL-fig")

    rows = []
    img_dir = output_dir / "aclfig_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    label_names = ds["train"].features["label"].names

    for split in ["train", "validation"]:
        for i, sample in enumerate(ds[split]):
            acl_label = label_names[sample["label"]]
            comp_label = ACLFIG_MAP.get(acl_label)
            if comp_label is None:
                continue
            img_path = img_dir / f"{split}_{i:05d}.jpg"
            if not img_path.exists():
                sample["image"].save(str(img_path))
            rows.append({
                "image_path": str(img_path),
                "label": comp_label,
                "source": "aclfig",
            })

    df = pd.DataFrame(rows)
    print(f"[ACL-Fig] {len(df)} usable samples from {len(ds['train'])+len(ds['validation'])} total")
    print(df["label"].value_counts().to_string())
    return df


def prepare_docfigure(docfigure_root: Path, output_dir: Path) -> pd.DataFrame:
    """DocFigure has flat images/ dir + annotation/train.txt CSV.

    annotation/train.txt format: "filename.png, Category Name"
    (category name has a leading space that we strip)
    """
    # Find annotation file
    annot_file = docfigure_root / "annotation" / "train.txt"
    if not annot_file.exists():
        candidates = list(docfigure_root.rglob("train.txt"))
        if candidates:
            annot_file = candidates[0]
        else:
            print(f"[DocFigure] Could not find annotation/train.txt in {docfigure_root}")
            return pd.DataFrame()

    # Find images directory
    images_dir = docfigure_root / "images"
    if not images_dir.exists():
        candidates = list(docfigure_root.rglob("images"))
        candidates = [c for c in candidates if c.is_dir()]
        if candidates:
            images_dir = candidates[0]
        else:
            print(f"[DocFigure] Could not find images/ in {docfigure_root}")
            return pd.DataFrame()

    rows = []
    skipped_cats = set()
    with open(annot_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: "filename.png, Category Name"
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            filename = parts[0].strip()
            cat_name = parts[1].strip()  # strip leading/trailing whitespace
            comp_label = DOCFIGURE_MAP.get(cat_name)
            if comp_label is None:
                skipped_cats.add(cat_name)
                continue
            img_path = images_dir / filename
            if not img_path.exists():
                continue
            rows.append({
                "image_path": str(img_path),
                "label": comp_label,
                "source": "docfigure",
            })

    if skipped_cats:
        print(f"[DocFigure] Skipped categories (unmapped/None): {sorted(skipped_cats)}")
    df = pd.DataFrame(rows)
    print(f"[DocFigure] {len(df)} usable samples")
    if len(df):
        print(df["label"].value_counts().to_string())
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aclfig", action="store_true")
    parser.add_argument("--docfigure", type=str, default=None, help="Path to extracted DocFigure dir")
    parser.add_argument("--output", type=str, default="data/train_augmented.csv")
    parser.add_argument("--external-dir", type=str, default="data/external")
    args = parser.parse_args()

    output_dir = Path(args.external_dir)
    dfs = []

    # Always include original training data
    orig = pd.read_csv("data/train_panels.csv")
    orig["source"] = "competition"
    dfs.append(orig)
    print(f"[Original] {len(orig)} samples")

    if args.aclfig:
        df = prepare_aclfig(output_dir)
        if len(df):
            dfs.append(df)

    if args.docfigure:
        df = prepare_docfigure(Path(args.docfigure), output_dir)
        if len(df):
            dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(args.output, index=False)
    print(f"\n[Combined] {len(combined)} total samples → {args.output}")
    print(combined["label"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
