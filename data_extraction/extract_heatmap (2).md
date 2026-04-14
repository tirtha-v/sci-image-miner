# Heatmap / Contour Chart Extraction Skill
## Claude Code Skill | ICDAR Task 2 | Version 2.0

---

## PURPOSE

Extract underlying data from heatmap and contour figures into structured Markdown tables and format the output as valid ICDAR Task 2 JSON entries.

This skill is called by the master SKILL.md when a heatmap-type chart is detected. It processes one image at a time within a batch loop.

---

## CHART SUBTYPES HANDLED BY THIS SKILL

| Subtype | Description |
|---|---|
| **Discrete Heatmap** | Grid of cells, each labeled with a numeric value |
| **Continuous Heatmap** | Color-encoded 2D map, values read from colorbar |
| **Contour Plot** | Iso-value lines over a 2D parameter space |
| **Filled Contour Plot** | Contour plot with color-filled regions |
| **Band Diagram** | Energy band levels vs position/material stack |
| **Process Timing Diagram** | ALD cycle timing: precursor/purge steps on a time axis |
| **Chromaticity Diagram** | CIE color space chart with coordinates |
| **Correlation Heatmap** | Matrix showing pairwise correlation coefficients |
| **Treemap** | Hierarchical area chart with labeled rectangular cells |
| **Process Window Map** | 2D map of process parameter space showing viable regions |

---

## CONTEXT

- **Task**: ICDAR 2026 Sci-ImageMiner Competition, Task 2
- **Domain**: Atomic Layer Deposition / Etching (ALD/E) scientific figures
- **Evaluation**: TED (Tree Edit Distance) + Relative Mapping Similarity
- **Key principle**: Always extract something — even an approximate table scores better than empty

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
- Look for clear visual separation between sub-plots
- If multiple panels → extract each separately, label "a", "b", "c" etc.
- If single panel → use "a" as the only key

---

## REASONING STEPS

### Step 1 — Identify subtype
Which subtype from the list above is this?

### Step 2 — Identify axes and colorbar
- X axis: label, unit, tick values
- Y axis: label, unit, tick values
- Colorbar: min value, max value, unit, label

### Step 3 — Extract data by subtype

**Discrete heatmap:**
- Matrix-style table: first column = Y axis labels, remaining columns = X axis tick values
- Cell values = numbers shown in each cell
- `| Y\X | X1 | X2 | X3 |`

**Continuous heatmap:**
- If values are color-only → sample key points (corners, center, notable regions)
- Estimate values from colorbar scale
- `| X (unit) | Y (unit) | Value (unit) |`

**Contour plot:**
- One row per labeled contour line
- `| X (unit) | Y (unit) | Contour Level (unit) |`
- Focus on labeled contours — approximate unlabeled ones

**Band diagram:**
- `| Position/Material | Energy Level (eV) | Band Type |`
- Band type = conduction band, valence band, Fermi level, etc.
- One row per labeled level

**Process timing diagram:**
- `| Step | Precursor/Gas | Start (s) | End (s) | Duration (s) |`
- One row per step in the ALD cycle

**Correlation heatmap:**
- Matrix table with variable names as both row and column headers
- Cell values = correlation coefficients

**Process window map:**
- `| X Parameter (unit) | Y Parameter (unit) | Region/Value |`
- One row per labeled region or notable point

### Step 4 — Handle colorbar-only values
- Note colorbar range in header: `Value (unit, colorbar: min–max)`
- Flag estimated values with ~ prefix

### Step 5 — Check completeness
- Discrete heatmap: row × column count matches grid?
- Contour: all labeled contour lines have entries?

---

## OUTPUT FORMAT

```json
{
  "sample_id": "atomic-layer-deposition/experimental-usecase/1/figure_2",
  "_confidence": "high",
  "_chart_type": "heatmap",
  "data_extraction": {
    "a": "```markdown\n| Y\\X | X1 | X2 | X3 |\n| --- | --- | --- | --- |\n| Y1 | val | val | val |\n| Y2 | val | val | val |\n```"
  }
}
```

Rules:
- Every table wrapped in ` ```markdown ` and ` ``` ` markers
- Standard GitHub Markdown table syntax
- Separator row: `| --- |`
- No extractable data → `"```markdown\n| No extractable data |\n| --- |\n```"`
- No text outside the JSON block
- `_confidence`: `high` / `medium` / `low`
- `_chart_type`: always `heatmap` for this skill
