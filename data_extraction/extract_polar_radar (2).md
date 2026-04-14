# Polar / Radar Chart Extraction Skill
## Claude Code Skill | ICDAR Task 2 | Version 2.0

---

## PURPOSE

Extract underlying data from polar and radar chart figures into structured Markdown tables and format the output as valid ICDAR Task 2 JSON entries.

This skill is called by the master SKILL.md when a polar_radar-type chart is detected. It processes one image at a time within a batch loop.

---

## CHART SUBTYPES HANDLED BY THIS SKILL

| Subtype | Description |
|---|---|
| **Polar Chart** | Data plotted on circular axes with angle and radius |
| **Rose Chart** | Polar histogram — sectors of varying radius |
| **Radar Chart** | Multiple axes radiating from center, one per variable |
| **Spider Chart** | Same as radar chart, polygon shape connecting axis values |

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
- If multiple panels → extract each separately, label "a", "b", "c" etc.
- If single panel → use "a" as the only key

---

## REASONING STEPS

### Step 1 — Identify subtype
**Radar / Spider:**
- Multiple axes radiating from center
- Each axis = a different variable/property
- One or more polygons = different series

**Polar / Rose:**
- Circular chart with angular axis (degrees or categories)
- Radial axis = magnitude or frequency

### Step 2 — Identify scale
- What is the radial scale? (min, max, tick intervals)
- What are the axis labels / angular categories?
- How many series are there? (from legend)

### Step 3 — Extract data by subtype

**Radar / Spider:**
- One row per axis
- One column per series
- `| Property (scale range) | Series 1 | Series 2 |`
- Estimate values by judging how far each polygon vertex extends along the axis relative to scale

**Polar / Rose:**
- One row per segment or labeled point
- Go clockwise from 0° / 12 o'clock position
- `| Angle / Category | Value (unit) |`

### Step 4 — Flag estimated values
- Radar values are almost always estimated from visual position
- Use ~ prefix for estimated values: `~7.5`

### Step 5 — Check completeness
- Radar: rows = number of axes in chart
- Polar/Rose: rows = number of segments or labeled points

---

## OUTPUT FORMAT

```json
{
  "sample_id": "atomic-layer-deposition/experimental-usecase/1/figure_2",
  "_confidence": "medium",
  "_chart_type": "polar_radar",
  "data_extraction": {
    "a": "```markdown\n| Property | Series 1 | Series 2 |\n| --- | --- | --- |\n| val | val | val |\n```"
  }
}
```

Rules:
- Every table wrapped in ` ```markdown ` and ` ``` ` markers
- Standard GitHub Markdown table syntax
- Separator row: `| --- |`
- No extractable data → `"```markdown\n| No extractable data |\n| --- |\n```"`
- No text outside the JSON block
- `_confidence`: `high` / `medium` / `low` (default `medium` — values are usually estimated)
- `_chart_type`: always `polar_radar` for this skill
