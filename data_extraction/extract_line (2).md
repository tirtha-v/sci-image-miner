# Line Chart Extraction Skill
## Claude Code Skill | ICDAR Task 2 | Version 2.0

---

## PURPOSE

Extract underlying data from line chart figures into structured Markdown tables and format the output as valid ICDAR Task 2 JSON entries.

This skill is called by the master SKILL.md when a line-type chart is detected. It processes one image at a time within a batch loop.

---

## CHART SUBTYPES HANDLED BY THIS SKILL

| Subtype | Description |
|---|---|
| **Single Line Chart** | One line, two axes, continuous data |
| **Multiple Line Chart** | Two or more lines on same axes, each from legend |
| **Multi-axis Chart** | Two Y axes (left and right) with different units |
| **Step-function Plot** | Discrete steps, common in QCM/ALD cycle traces |
| **Saturation Curve** | Asymptotic curve approaching plateau (precursor dose curves) |
| **GPC vs Temperature Curve** | Growth per cycle plotted against substrate temperature |
| **Spectra Chart** | XPS, XRD, FTIR — intensity vs energy/wavenumber/angle |
| **Stacked Spectra Chart** | Multiple spectra offset vertically for comparison |
| **Multi Spectra Chart** | Multiple spectra on same axes |
| **Area Chart** | Lines with shaded area beneath |
| **Competing Reaction Rate Curve** | Multiple reaction rate lines vs a parameter |
| **Reaction Energy Profile** | Energy vs reaction coordinate with labeled states |

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
Which subtype from the list above is this? This determines the extraction approach.

### Step 2 — Identify axes
- X axis: label, unit, range, tick values
- Y axis: label, unit, range, tick values
- Secondary Y axis (if present): label and unit

### Step 3 — Identify data series
- How many lines/series?
- Name of each series (from legend or annotation)
- Color/style of each series

### Step 4 — Extract data points
For each series:
- Read every clearly visible data point
- **Step-function**: capture every transition (both before and after each step)
- **Saturation curve**: capture plateau + all approach points
- **Spectra**: capture all peak positions and their intensities
- **Multi-axis**: extract both Y axes — use separate columns with distinct headers
- If exact values are hard to read → estimate from axis scale (approximate > missing)
- Note error bars with ± column if present

### Step 5 — Structure the table
- Multiple series, same X values → one table, one X column, one column per series
- Multiple series, different X values → separate tables per series, labeled by panel
- Always include units in headers: `Temperature (°C)` not `Temperature`

### Step 6 — Check completeness
- Row count matches visible data points?
- All series from legend represented?
- Units in all headers?

---

## OUTPUT FORMAT

```json
{
  "sample_id": "atomic-layer-deposition/experimental-usecase/1/figure_2",
  "_confidence": "high",
  "_chart_type": "line",
  "data_extraction": {
    "a": "```markdown\n| X Header (unit) | Series 1 (unit) | Series 2 (unit) |\n| --- | --- | --- |\n| val | val | val |\n```",
    "b": "```markdown\n| X Header (unit) | Y Header (unit) |\n| --- | --- |\n| val | val |\n```"
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
- `_chart_type`: always `line` for this skill
