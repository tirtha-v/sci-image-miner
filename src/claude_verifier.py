"""Targeted Claude API calls for hybrid pipeline verification.

Two functions:
1. disambiguate_labels() — text-only Haiku call for confusable classification pairs
2. enrich_vqa_paragraphs() — Sonnet call to ground paragraph answers in extraction data
"""

import json
import os
import re
from typing import Optional

import anthropic

# Confusable pair groups — only panels predicted into these trigger Claude check
CONFUSABLE_GROUPS = [
    {"line chart", "multiple line chart"},
    {"scatter plot", "multiple scatter plot"},
    {"spectra chart", "stacked spectra chart", "multi spectra chart"},
    {"conceptual diagram", "apparatus diagram", "process flow diagram"},
    {"reaction scheme", "reaction energy profile diagram"},
    {"bar chart", "grouped bar chart", "stacked bar chart"},
]

# Phrases in raw output that suggest ambiguity
AMBIGUITY_PHRASES = [
    " or ", " possibly ", " could be ", " might be ", " either ",
    " unclear ", " unsure ", " difficult to ", " hard to ",
    " looks like both", " resembles both",
]

_CLIENT: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = anthropic.Anthropic()
    return _CLIENT


def _in_confusable_group(label: str) -> Optional[set]:
    label_lower = label.lower().strip()
    for group in CONFUSABLE_GROUPS:
        if label_lower in group:
            return group
    return None


def _is_ambiguous(raw_output: str, label: str) -> bool:
    """Return True if raw output suggests the model was uncertain."""
    raw_lower = raw_output.lower()
    for phrase in AMBIGUITY_PHRASES:
        if phrase in raw_lower:
            return True
    # Also trigger if label is in a confusable group AND raw output mentions another group member
    group = _in_confusable_group(label)
    if group:
        for other_label in group:
            if other_label != label.lower() and other_label in raw_lower:
                return True
    return False


def disambiguate_labels(
    panels: list[dict],
    taxonomy: list[str],
) -> dict[str, dict[str, str]]:
    """
    Disambiguate classification labels using Claude Haiku (text-only, no images).

    Args:
        panels: list of {"sample_id": str, "panel_id": str, "label": str, "raw": str}
        taxonomy: full list of 49 class names

    Returns:
        dict mapping (sample_id, panel_id) → corrected label string
        Only includes panels that were actually sent to Claude for correction.
    """
    client = _get_client()

    # Filter to only ambiguous panels in confusable groups
    to_disambiguate = []
    for p in panels:
        if _is_ambiguous(p.get("raw", ""), p.get("label", "")):
            group = _in_confusable_group(p.get("label", ""))
            if group:
                to_disambiguate.append({**p, "candidates": sorted(group)})

    if not to_disambiguate:
        return {}

    print(f"[claude_verifier] Disambiguating {len(to_disambiguate)} panels via Haiku ...")

    corrections = {}
    # Batch 20 panels per call
    batch_size = 20
    for i in range(0, len(to_disambiguate), batch_size):
        batch = to_disambiguate[i:i + batch_size]
        corrections.update(_disambiguate_batch(client, batch))

    return corrections


def _disambiguate_batch(client: anthropic.Anthropic, batch: list[dict]) -> dict:
    items_text = ""
    for j, p in enumerate(batch):
        items_text += (
            f"\n--- Panel {j+1} ---\n"
            f"Model output: {p['raw'][:500]}\n"
            f"Candidate labels: {', '.join(p['candidates'])}\n"
        )

    prompt = (
        "You are an expert in classifying scientific figures from materials science papers "
        "about atomic layer deposition (ALD) and atomic layer etching (ALE).\n\n"
        "For each panel below, a vision model gave an ambiguous classification. "
        "Based ONLY on the model's description of the image, pick the single most likely label "
        "from the provided candidates.\n\n"
        "Respond with a JSON array with one entry per panel: "
        '[{"panel": 1, "label": "chosen label"}, ...]\n\n'
        f"{items_text}"
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        # Extract JSON array
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if not match:
            return {}
        parsed = json.loads(match.group())
        result = {}
        for item in parsed:
            idx = item.get("panel", 0) - 1
            if 0 <= idx < len(batch):
                p = batch[idx]
                label = item.get("label", "").strip().lower()
                if label in p["candidates"]:
                    result[(p["sample_id"], p["panel_id"])] = label
        return result
    except Exception as e:
        print(f"[claude_verifier] Haiku batch error: {e}")
        return {}


def enrich_vqa_paragraphs(
    panels: list[dict],
) -> dict[tuple, list[dict]]:
    """
    Enrich Paragraph-type VQA answers using extraction table data via Claude Sonnet.

    Args:
        panels: list of {
            "sample_id": str, "panel_id": str,
            "classification": str,
            "extraction": str,   # markdown table or empty string
            "summary": str,      # panel summary or empty string
            "qa_list": [{"question_type", "question", "answer_type", "answer"}]
        }

    Returns:
        dict mapping (sample_id, panel_id) → enriched qa_list
        Only includes panels with Paragraph answers that have non-empty extraction data.
    """
    client = _get_client()

    # Filter to panels with Paragraph answers AND non-empty extraction
    to_enrich = []
    for p in panels:
        has_paragraph = any(
            qa.get("answer_type") == "Paragraph"
            for qa in p.get("qa_list", [])
        )
        has_extraction = bool(p.get("extraction", "").strip())
        if has_paragraph and has_extraction:
            to_enrich.append(p)

    if not to_enrich:
        return {}

    print(f"[claude_verifier] Enriching VQA paragraphs for {len(to_enrich)} panels via Sonnet ...")

    enriched = {}
    # Batch 10 panels per call
    batch_size = 10
    for i in range(0, len(to_enrich), batch_size):
        batch = to_enrich[i:i + batch_size]
        enriched.update(_enrich_batch(client, batch))

    return enriched


def _enrich_batch(client: anthropic.Anthropic, batch: list[dict]) -> dict:
    items_text = ""
    for j, p in enumerate(batch):
        paragraph_questions = [
            qa for qa in p.get("qa_list", [])
            if qa.get("answer_type") == "Paragraph"
        ]
        items_text += (
            f"\n--- Panel {j+1} ({p['sample_id']}, panel {p['panel_id']}) ---\n"
            f"Figure type: {p.get('classification', 'unknown')}\n"
            f"Extracted data:\n{p.get('extraction', '')}\n"
            f"Summary: {p.get('summary', '')}\n"
            "Questions to enrich:\n"
        )
        for k, qa in enumerate(paragraph_questions):
            items_text += (
                f"  Q{k+1}: {qa['question']}\n"
                f"  Draft answer: {qa['answer']}\n"
            )

    prompt = (
        "You are an expert in materials science specializing in ALD and ALE.\n\n"
        "For each panel below, improve the draft answers to the Paragraph-type questions "
        "by incorporating specific quantitative values from the extracted data table. "
        "Keep answers concise (2-4 sentences). Do not add information not present in the data.\n\n"
        "Respond with a JSON array:\n"
        '[{"panel": 1, "answers": ["improved answer for Q1", "improved answer for Q2", ...]}, ...]\n\n'
        f"{items_text}"
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if not match:
            return {}
        parsed = json.loads(match.group())

        result = {}
        for item in parsed:
            idx = item.get("panel", 0) - 1
            if 0 <= idx < len(batch):
                p = batch[idx]
                improved_answers = item.get("answers", [])
                paragraph_indices = [
                    i for i, qa in enumerate(p["qa_list"])
                    if qa.get("answer_type") == "Paragraph"
                ]
                new_qa_list = list(p["qa_list"])  # copy
                for ans_idx, qa_idx in enumerate(paragraph_indices):
                    if ans_idx < len(improved_answers) and improved_answers[ans_idx]:
                        new_qa_list[qa_idx] = {
                            **new_qa_list[qa_idx],
                            "answer": improved_answers[ans_idx],
                        }
                result[(p["sample_id"], p["panel_id"])] = new_qa_list
        return result
    except Exception as e:
        print(f"[claude_verifier] Sonnet batch error: {e}")
        return {}
