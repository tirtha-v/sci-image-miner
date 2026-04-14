# Scatter Chart Extraction Skill
## Claude Code Skill | ICDAR Task 2 | Version 2.0

---

## PURPOSE

Extract underlying data from scatter and point-based chart figures into structured Markdown tables and format the output as valid ICDAR Task 2 JSON entries.

This skill is called by the master SKILL.md when a scatter-type chart is detected. It processes one image at a time within a batch loop.

---

## CHART SUBTYPES HANDLED BY THIS SKILL

| Subtype | Description |
|---|---|
| **Simple Scatter Plot** | X-Y points with no connecting line |
| **Multiple Scatter Plot** | Multiple series of X-Y points, each from legend |
| **Scatter with Fitted Curve** | Scatter points with regression/fit line overlaid |
| **3D Scatter Plot** | Points in 3D space with X, Y, Z axes |
| **Bubble Chart** | Scatter where point size encodes a third variable |
| **Box Plot** | Statistical distribution: min, Q1, median, Q3, max per category |
| **Violin Plot** | Distribution shape shown as mirrored density |
| **Strip Plot** | Individual points per category (like a box plot with raw data) |
| **Phase Diagram (points)** | Discrete data points marking phase boundaries |
| **Arrhenius Plot** | ln(rate) vs 1/T scatter, common in ALD kinetics |

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

### Step 2 — Identify axes
- X axis: label, unit, range, scale (linear or log?)
- Y axis: label, unit, range, scale (linear or log?)
- Z axis / bubble size axis if present

### Step 3 — Extract data by subtype

**Simple / Multiple scatter:**
- One row per data point
- Multiple series with same X → one table, one X column, one Y column per series
- Multiple series with different X → separate tables per series

**Scatter with fitted curve:**
- Extract raw scatter points only
- Note fit type (linear, exponential, power law) in a header note if labeled

**3D scatter:**
- `| X (unit) | Y (unit) | Z (unit) |`
- One row per point

**Bubble chart:**
- `| X (unit) | Y (unit) | Size (unit) |`
- Note what bubble size represents from legend

**Box plot:**
- `| Category | Min | Q1 | Median | Q3 | Max |`
- One row per box
- Add outlier column if outliers are marked

**Violin / Strip plot:**
- Extract visible summary statistics if labeled
- `| Category | Value |` for strip plots

**Arrhenius plot:**
- `| 1/T (K⁻¹) | ln(rate) |` or use axis labels as-is
- Note activation energy if annotated on the plot

### Step 4 — Handle error bars
- Error bars present → add `± Error (unit)` column

### Step 5 — Check completeness
- Count visible points — does row count match?
- All legend series represented?

---

## OUTPUT FORMAT

```json
{
  "sample_id": "atomic-layer-deposition/experimental-usecase/1/figure_2",
  "_confidence": "high",
  "_chart_type": "scatter",
  "data_extraction": {
    "a": "```markdown\n| X Header (unit) | Y Header (unit) |\n| --- | --- |\n| val | val |\n```"
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
- `_chart_type`: always `scatter` for this skill
