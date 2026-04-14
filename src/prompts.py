"""Taxonomy and prompt templates for chart classification."""

import os
from pathlib import Path

# Resolve taxonomy file relative to the repo root
_REPO_ROOT = Path(__file__).parent.parent / "ALD-E-ImageMiner"
_TAXONOMY_TSV = _REPO_ROOT / "figure_taxonomy.tsv"


def _load_taxonomy(tsv_path: Path) -> list[str]:
    """Load the 49-label taxonomy from the TSV file (first column, skip header)."""
    labels = []
    with open(tsv_path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # skip header
            parts = line.strip().split("\t")
            if parts and parts[0]:
                labels.append(parts[0].strip())
    return labels


def _load_taxonomy_with_descriptions(tsv_path: Path) -> list[tuple[str, str]]:
    """Load (label, short_description) pairs from the TSV file."""
    entries = []
    with open(tsv_path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # skip header
            parts = line.strip().split("\t")
            if parts and parts[0]:
                label = parts[0].strip()
                desc = parts[1].strip() if len(parts) > 1 else ""
                entries.append((label, desc))
    return entries


TAXONOMY: list[str] = _load_taxonomy(_TAXONOMY_TSV)
_TAXONOMY_WITH_DESC: list[tuple[str, str]] = _load_taxonomy_with_descriptions(_TAXONOMY_TSV)

_TAXONOMY_STR = ", ".join(TAXONOMY)
_TAXONOMY_DESC_STR = "\n".join(
    f"- {label}: {desc}" if desc else f"- {label}"
    for label, desc in _TAXONOMY_WITH_DESC
)

# Selective hints — only for classes that are commonly confused with each other.
# All other classes use their plain name (they were already performing well).
_SELECTIVE_HINTS: dict[str, str] = {
    # Line/scatter family — single vs multiple series is the key distinction
    "line chart": "SINGLE continuous line/curve — one data series only",
    "multiple line chart": "TWO OR MORE separate lines/curves on the same axes — multiple data series",
    "scatter plot": "x-y data as DISCRETE POINTS or MARKERS (dots, circles, triangles); may include a single fit/trend curve; SINGLE group or dataset",
    "multiple scatter plot": "x-y data as DISCRETE POINTS from MULTIPLE groups/series on the same axes, distinguished by color, shape, or marker type",
    "multi-axis chart": "Plot with two or more y-axes at DIFFERENT scales (dual-axis / twin-axis)",
    # Spectra family — key is single vs stacked vs overlaid
    "spectra chart": "SINGLE spectrum — one curve of intensity vs wavelength/energy/wavenumber (NMR, IR, Raman, XRD, UV-vis, MS)",
    "stacked spectra chart": "Multiple spectra arranged VERTICALLY STACKED (offset along y-axis) so each spectrum sits above the previous",
    "multi spectra chart": "Multiple spectra OVERLAID on the SAME axes (all on one plot, not stacked, no vertical offset)",
    # Timing / process
    "process timing diagram": "Time-axis showing gas pulse/purge sequences as rectangular step-like waveforms over an ALD/ALE half-cycle",
    # Conceptual — easily confused with apparatus or process flow
    "conceptual diagram": "Illustrative schematic explaining a CONCEPT or MECHANISM — not a data plot, not a chemical structure, not an apparatus photo",
    # Reaction family
    "reaction scheme": "Arrows and molecule structures showing a CHEMICAL REACTION pathway — reactants transforming to products",
    "reaction energy profile diagram": "Y-axis is ENERGY (or free energy), x-axis is reaction coordinate — shows transition states and intermediates",
}

_TAXONOMY_SELECTIVE_STR = "\n".join(
    f"- {label}: {_SELECTIVE_HINTS[label]}" if label in _SELECTIVE_HINTS else f"- {label}"
    for label in TAXONOMY
)

SYSTEM_PROMPT = (
    "You are an expert at classifying scientific figures from materials science papers "
    "about atomic layer deposition (ALD) and atomic layer etching (ALE). "
    "Your task is to identify the chart or figure type shown in an image."
)

USER_PROMPT_TEMPLATE = (
    "Classify this scientific figure into exactly ONE category from the list below.\n"
    "Output ONLY the category name, nothing else — no punctuation, no explanation.\n\n"
    "Categories: {taxonomy}\n\n"
    "Category:"
)

USER_PROMPT_TEMPLATE_WITH_DESC = (
    "Classify this scientific figure into exactly ONE category from the list below.\n"
    "Output ONLY the category name exactly as written, nothing else — no punctuation, no explanation.\n\n"
    "Categories:\n{taxonomy}\n\n"
    "Category:"
)


def build_classification_prompt(
    few_shot: bool = False,
    with_descriptions: bool = False,
    selective_descriptions: bool = False,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for classification.

    Args:
        few_shot: If True, include text exemplars in the prompt
        with_descriptions: If True, include short descriptions for all categories
        selective_descriptions: If True, add targeted hints only for confusable classes
    """
    if selective_descriptions:
        user_prompt = USER_PROMPT_TEMPLATE_WITH_DESC.format(taxonomy=_TAXONOMY_SELECTIVE_STR)
    elif with_descriptions:
        user_prompt = USER_PROMPT_TEMPLATE_WITH_DESC.format(taxonomy=_TAXONOMY_DESC_STR)
    else:
        user_prompt = USER_PROMPT_TEMPLATE.format(taxonomy=_TAXONOMY_STR)

    if few_shot:
        user_prompt = (
            "Here are some examples of how to classify scientific figures:\n\n"
            "Example 1: A figure showing XPS survey spectra with binding energy on x-axis "
            "and intensity/counts on y-axis, with multiple labeled peaks → spectra chart\n\n"
            "Example 2: A figure with multiple XRD patterns stacked vertically, each at "
            "different temperatures, with 2θ on x-axis → stacked spectra chart\n\n"
            "Example 3: A figure showing GPC (growth per cycle) vs temperature with data "
            "points connected by lines → line chart\n\n"
            "Example 4: SEM/TEM/AFM microscopy images arranged in a grid → image panel\n\n"
            "Example 5: A 2D chemical structure drawing of a precursor molecule → "
            "molecular structure diagram\n\n"
            "Example 6: A diagram showing the ALD reactor chamber with substrate, "
            "gas inlets, and heating elements → apparatus diagram\n\n"
            "Example 7: A figure with energy on y-axis and reaction coordinate on x-axis, "
            "showing transition states and intermediates → reaction energy profile diagram\n\n"
            "Example 8: A color-coded map of film thickness or composition across a "
            "substrate surface → contour heatmap\n\n"
            "Example 9: A scatter plot with multiple series differentiated by color/shape → "
            "multiple scatter plot\n\n"
            "Example 10: A timing sequence showing precursor/reactant pulse and purge "
            "durations → process timing diagram\n\n"
            "Now classify the following figure:\n\n"
        ) + user_prompt

    return SYSTEM_PROMPT, user_prompt
