# Table Extraction Skill
## Claude Code Skill | ICDAR Task 2 | Version 2.0

---

## PURPOSE

Extract data from tabular figures into structured Markdown tables and format the output as valid ICDAR Task 2 JSON entries.

This skill is called by the master SKILL.md when a table-type figure is detected. It processes one image at a time within a batch loop.

---

## CHART SUBTYPES HANDLED BY THIS SKILL

| Subtype | Description |
|---|---|
| **Simple Data Table** | Rows and columns with a header row, plain data |
| **Comparison Table** | Side-by-side comparison of methods, materials, or conditions |
| **Element-property Matrix** | Elements/materials as rows, properties as columns |
| **Periodic Table Map** | Subset of periodic table with property values per element |
| **Process Parameter Table** | ALD recipe parameters: temperature, dose, purge time etc. |
| **Results Summary Table** | Tabulated experimental results across conditions |
| **Multi-level Header Table** | Nested or grouped column headers |
| **Mixed Units Table** | Each column may have different units in the header |

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
- Look for clear visual separation between sub-tables
- If multiple panels → extract each separately, label "a", "b", "c" etc.
- If single panel → use "a" as the only key

---

## REASONING STEPS

### Step 1 — Identify subtype
Which subtype from the list above is this?

### Step 2 — Identify structure
- How many columns?
- Is the first row a header?
- Are there merged cells? → expand into individual cells
- Are there multi-level headers? → flatten to single header row, combine parent+child: `Material | Thickness (nm)` becomes `Material - Thickness (nm)`

### Step 3 — Identify and construct header row
- Use first row if it contains labels, not values
- If no clear header → infer from context: `Parameter`, `Value`, `Unit`, `Condition`
- Always include units: `GPC (Å/cycle)` not `GPC`

### Step 4 — Extract all rows
- Extract every row exactly as written — no paraphrasing
- Preserve chemical formulas: Al₂O₃, TiO₂, H₂O, TMA, TDMAT
- Preserve numeric precision: `0.85` not `0.9`, `1.10` not `1.1`
- Empty cell → empty string in that position

### Step 5 — Handle special subtypes

**Comparison table:**
- First column = parameter being compared
- Remaining columns = materials/methods being compared

**Element-property matrix:**
- Row labels = elements or materials
- Column labels = properties with units

**Multi-level headers:**
- Flatten: `Precursor | Dose (s) | Purge (s)` — don't nest

### Step 6 — Check completeness
- Row count matches visible rows?
- Column count matches exactly?
- No cells silently dropped?

---

## OUTPUT FORMAT

```json
{
  "sample_id": "atomic-layer-deposition/experimental-usecase/1/figure_2",
  "_confidence": "high",
  "_chart_type": "table",
  "data_extraction": {
    "a": "```markdown\n| Col1 | Col2 (unit) | Col3 (unit) |\n| --- | --- | --- |\n| val | val | val |\n```"
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
- `_chart_type`: always `table` for this skill
