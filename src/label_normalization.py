"""Label cleaning: fix obvious typos/spelling errors in training labels."""

# Only fix obvious typos/spelling errors — do NOT remap categories
LABEL_FIXES = {
    "unknmolecular structure diagramwn": "unknown",
    "apparatus diagra": "apparatus diagram",
    "molecular structure diagramm": "molecular structure diagram",
    "device structure diagram": "device structure",
}


def clean_label(raw_label: str) -> str:
    """Fix known typos in labels. Returns the cleaned label."""
    cleaned = raw_label.strip().lower()
    return LABEL_FIXES.get(cleaned, raw_label.strip())
