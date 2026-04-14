"""Post-processing: normalize raw VLM output to a taxonomy label."""

import difflib

from .label_cleaning import clean_label


def normalize_label(raw_output: str, taxonomy: list[str]) -> str:
    """Map raw model output to the closest taxonomy label.

    Strategy:
        0. Fix known typos via label_cleaning
        1. Exact match (case-insensitive, stripped)
        2. Substring match
        3. Fuzzy match via difflib (cutoff=0.5)
        4. Fallback to "unknown"
    """
    raw = clean_label(raw_output).strip().lower()

    # 1. Exact match
    for label in taxonomy:
        if label.lower() == raw:
            return label

    # 2. Substring match (model sometimes outputs just part of the label)
    for label in taxonomy:
        if label.lower() in raw or raw in label.lower():
            return label

    # 3. Fuzzy match
    lower_taxonomy = [t.lower() for t in taxonomy]
    matches = difflib.get_close_matches(raw, lower_taxonomy, n=1, cutoff=0.5)
    if matches:
        idx = lower_taxonomy.index(matches[0])
        return taxonomy[idx]

    # 4. Fallback
    return "unknown"
