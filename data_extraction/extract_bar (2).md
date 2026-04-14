# Bar Chart Extraction Skill
## Claude Code Skill | ICDAR Task 2 | Version 2.0

---

## PURPOSE

Extract underlying data from bar and categorical chart figures into structured Markdown tables and format the output as valid ICDAR Task 2 JSON entries.

This skill is called by the master SKILL.md when a bar-type chart is detected. It processes one image at a time within a batch loop.

---

## CHART SUBTYPES HANDLED BY THIS SKILL

| Subtype | Description |
|---|---|
| **Simple Bar Chart** | Single bars per category, one value per bar |
| **Grouped Bar Chart** | Multiple bars per category group, one color per series |
| **Stacked Bar Chart** | Bars divided into segments stacked on top of each other |
| **3D Bar Chart** | Bar chart rendered in 3D perspective |
| **Horizontal Bar Chart** | Bars extending horizontally instead of vertically |
| **Pie Chart** | Circular chart divided into proportional slices |
| **Donut Chart** | Pie chart with hollow center |
| **Funnel Chart** | Tapering chart showing sequential stages |
| **Histogram** | Bar chart showing frequency distribution |

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
Which subtype from the list above is this? This determines the table structure.

### Step 2 — Identify axes and categories
- Category axis label (usually X)
- Value axis label and unit (usually Y)
- All category labels listed on axis

### Step 3 — Extract values by subtype

**Simple bar:**
- One row per bar: `| Category | Value (unit) |`

**Grouped bar:**
- One row per category group
- One column per group member (from legend)
- `| Category | Group1 (unit) | Group2 (unit) |`

**Stacked bar:**
- One row per category
- One column per stack segment (from legend)
- Include total column if visible
- `| Category | Segment1 (unit) | Segment2 (unit) | Total (unit) |`

**Horizontal bar:**
- Same as simple/grouped but note axis orientation in headers

**Pie / Donut:**
- `| Label | Value (%) |` or `| Label | Value (unit) |`
- One row per slice, ordered by size descending

**Funnel:**
- `| Stage | Value |`
- Rows in funnel order top to bottom

**Histogram:**
- `| Bin Range | Frequency |`
- One row per bar

### Step 4 — Handle value labels
- Labels printed on bars → use exact values
- No labels → estimate from axis scale
- Error bars present → add `± Error` column

### Step 5 — Check completeness
- Row count = number of bars/slices/categories?
- All legend groups as columns?
- Units in all value headers?

---

## OUTPUT FORMAT

```json
{
  "sample_id": "atomic-layer-deposition/experimental-usecase/1/figure_2",
  "_confidence": "high",
  "_chart_type": "bar",
  "data_extraction": {
    "a": "```markdown\n| Category | Value (unit) |\n| --- | --- |\n| val | val |\n```",
    "b": "```markdown\n| Category | Group1 (unit) | Group2 (unit) |\n| --- | --- | --- |\n| val | val | val |\n```"
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
- `_chart_type`: always `bar` for this skill
