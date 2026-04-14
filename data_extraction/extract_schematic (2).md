# Schematic / Non-quantitative Figure Extraction Skill
## Claude Code Skill | ICDAR Task 2 | Version 2.0

---

## PURPOSE

Extract whatever structured information is visible from non-quantitative figures into Markdown tables and format the output as valid ICDAR Task 2 JSON entries.

These figures have no data axes — but labels, annotations, chemical names, process steps, and measurements are still extractable and will improve TED score over an empty submission.

This skill is called by the master SKILL.md when a schematic-type figure is detected. It processes one image at a time within a batch loop.

---

## CHART SUBTYPES HANDLED BY THIS SKILL

| Subtype | Description |
|---|---|
| **Apparatus Diagram** | Reactor setup, chamber, gas lines, labeled components |
| **Device Structure** | Layer stack cross-section with labeled materials and thicknesses |
| **Reaction Scheme** | Chemical reaction with reactants, products, arrows, conditions |
| **Molecular Structure Diagram** | 2D/3D molecular structure with labeled atoms/bonds |
| **Process Flow Diagram** | Sequence of process steps with arrows |
| **Conceptual Diagram** | Abstract illustration of a concept or mechanism |
| **Workflow Diagram** | Decision trees, pipelines, experimental procedures |
| **Timeline Chart** | Events or steps plotted along a time axis |
| **Network Diagram** | Nodes and edges representing relationships |
| **Tree Diagram** | Hierarchical branching structure |
| **Image Panel** | Microscopy or photography images (SEM, TEM, AFM, optical) |
| **Formula / Equation** | Mathematical or chemical formula as a figure |
| **Schematic Cross-section** | Cross-section of a device or material stack |

---

## CONTEXT

- **Task**: ICDAR 2026 Sci-ImageMiner Competition, Task 2
- **Domain**: Atomic Layer Deposition / Etching (ALD/E) scientific figures
- **Evaluation**: TED (Tree Edit Distance) + Relative Mapping Similarity
- **Key principle**: Never output nothing — structured text always scores better than empty on TED

---

## INPUT

This skill is invoked per-image by the master SKILL.md. It receives:
- `IMAGE_PATH`: full path to the current image being processed
- `SAMPLE_ID`: pre-constructed sample_id from the master skill

---

## SAMPLE ID CONSTRUCTION

Constructed by the master SKILL.md from the image file path.

Rule: take everything after the `test` folder segment, replace separators with `/`, drop file extension, drop any `images` folder segment.

Example:
```
Input:  .../test/atomic-layer-deposition/experimental-usecase/1/images/figure_2.jpg
Output: atomic-layer-deposition/experimental-usecase/1/figure_2
```

If no `test` segment exists in the path → use the filename stem as the sample_id.


## PANEL DETECTION

Before extracting, examine the image for multiple panels:
- Look for labels like (a), (b), (c) or A, B, C
- Look for clear visual separation between sub-figures
- If multiple panels → extract each separately, label "a", "b", "c" etc.
- If single panel → use "a" as the only key

---

## REASONING STEPS

### Step 1 — Identify subtype
Which subtype from the list above is this?

### Step 2 — Scan for structured information
Look for any of the following:
- Chemical names or formulas (Al₂O₃, TMA, H₂O, TDMAT, precursor names)
- Labeled dimensions or measurements (thickness in nm, temperature in °C)
- Process step names or sequence labels
- Material layer names and properties
- Reaction arrows with labeled reactants/products
- Annotated components (substrate, electrode, gate, chamber parts)
- Any numeric values with units

### Step 3 — Choose table structure by subtype

**Apparatus / device diagrams:**
`| Component | Label | Material | Notes |`

**Device structure / layer stack:**
`| Layer | Material | Thickness (nm) | Position |`

**Reaction scheme:**
`| Step | Reactants | Products | Conditions |`

**Molecular structure:**
`| Component | Chemical Formula | Role |`

**Process flow / workflow:**
`| Step | Label | Description |`

**Timeline:**
`| Event | Time | Description |`

**Formula / equation:**
`| Parameter | Symbol | Value | Unit |`

**Image panel (SEM/TEM/AFM):**
`| Panel | Scale Bar | Feature | Description |`

**Generic fallback (if none above fit):**
`| Element | Description |`

### Step 4 — Extract all visible text
- Preserve chemical formulas exactly
- Preserve all numeric values and units exactly
- Use exact labels from the figure — do not paraphrase
- Partially visible or ambiguous text → include best reading

### Step 5 — If truly nothing extractable
- Use: `| No extractable data |\n| --- |`
- This should be very rare — almost every schematic has labeled text

---

## OUTPUT FORMAT

```json
{
  "sample_id": "atomic-layer-deposition/experimental-usecase/1/figure_2",
  "_confidence": "high",
  "_chart_type": "schematic",
  "data_extraction": {
    "a": "```markdown\n| Component | Label | Material |\n| --- | --- | --- |\n| val | val | val |\n```"
  }
}
```

Rules:
- Every table wrapped in ` ```markdown ` and ` ``` ` markers
- Standard GitHub Markdown table syntax
- Separator row: `| --- |`
- No text outside the JSON block
- `_confidence`: `high` / `medium` / `low`
- `_chart_type`: always `schematic` for this skill
