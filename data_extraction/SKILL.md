# ICDAR Extraction Master Skill
## Claude Code Skill | ICDAR Task 2 | Version 3.0 (General)

---

## PURPOSE

Master entry point for ICDAR Task 2 batch extraction. Given a folder of chart images, this skill:
1. Detects chart type for each image
2. Reads the appropriate sub-skill file
3. Extracts data into Markdown tables following sub-skill reasoning steps
4. Accumulates all results into a single `submission.json`
5. Saves incrementally every 10 images (checkpoint protection)

---

## BEFORE RUNNING — USER MUST PROVIDE

When you paste this skill into Claude Code, provide the following at the bottom:

```
IMAGE_FOLDER:   <path to folder containing all test images>
SKILLS_FOLDER:  <path to folder containing the skills\ sub-skill .md files>
OUTPUT_FOLDER:  <path to folder where submission.json and checkpoint.txt will be saved>
```

Example:
```
IMAGE_FOLDER:   /content/drive/MyDrive/icdar-test-images
SKILLS_FOLDER:  /content/drive/MyDrive/icdar-skill/skills
OUTPUT_FOLDER:  /content/drive/MyDrive/icdar-outputs
```

---

## SKILL FOLDER STRUCTURE

All files are in the same flat folder (no subfolder):

```
data_extraction/
├── SKILL.md                        ← this file
├── extract_line (2).md
├── extract_bar (2).md
├── extract_scatter (2).md
├── extract_table (2).md
├── extract_heatmap (2).md
├── extract_schematic (2).md
└── extract_polar_radar (2).md
```

---

## SAMPLE ID CONSTRUCTION

For every image, construct the sample_id from its file path:

Rule: take everything after the `test` folder segment, replace `\` or `/` separators consistently, drop the file extension, drop any `images` folder segment.

Example:
```
Input:  .../test/atomic-layer-deposition/experimental-usecase/1/images/figure_2.jpg
Output: atomic-layer-deposition/experimental-usecase/1/figure_2
```

If no `test` folder segment exists in the path → use the filename stem as the sample_id.

---

## CHART TYPE ROUTING TABLE

| Detected chart type | Sub-skill file to read |
|---|---|
| Line, step-function, spectra, saturation curve, GPC curve, area | `extract_line (2).md` |
| Bar, grouped bar, stacked bar, pie, donut, funnel, histogram | `extract_bar (2).md` |
| Scatter, bubble, box plot, violin, strip, Arrhenius, phase diagram | `extract_scatter (2).md` |
| Table, comparison table, matrix, periodic table map | `extract_table (2).md` |
| Heatmap, contour, band diagram, process timing, correlation map | `extract_heatmap (2).md` |
| Schematic, apparatus, reaction scheme, device structure, flow diagram, SEM/TEM | `extract_schematic (2).md` |
| Polar, rose, radar, spider chart | `extract_polar_radar (2).md` |

---

## STEP-BY-STEP INSTRUCTIONS FOR CLAUDE CODE

### Step 1 — Setup
- Read the paths provided by the user at the bottom of this file
- Read all 7 sub-skill files from `SKILLS_FOLDER` into memory (files are named `extract_line (2).md`, `extract_bar (2).md`, `extract_scatter (2).md`, `extract_table (2).md`, `extract_heatmap (2).md`, `extract_schematic (2).md`, `extract_polar_radar (2).md`)
- Create `OUTPUT_FOLDER` if it does not exist
- Collect all image files (.jpg, .jpeg, .png, case-insensitive) recursively from `IMAGE_FOLDER`
- Sort files alphabetically for reproducibility
- Load `OUTPUT_FOLDER/checkpoint.txt` if it exists — skip sample_ids already listed
- Load `OUTPUT_FOLDER/submission.json` if it exists — append to it (resume mode)
- Print: total images found, already processed, remaining to process

### Step 2 — For each image: classify
Examine the image and classify into exactly one of these 7 types:
- `line` — any continuous line chart, step function, spectra, saturation curve, area chart
- `bar` — bar charts of any kind, pie, donut, funnel, histogram
- `scatter` — scatter plots, bubble charts, box plots, violin, Arrhenius
- `table` — data tables, comparison tables, matrices
- `heatmap` — heatmaps, contour plots, band diagrams, process timing
- `schematic` — non-quantitative figures, diagrams, apparatus, SEM/TEM images
- `polar_radar` — polar, rose, radar, spider charts

For multi-panel figures → classify by the dominant chart type.

### Step 3 — For each image: read sub-skill and extract
- Based on detected chart type, read the corresponding sub-skill from memory
- Follow its REASONING STEPS exactly to extract the data
- Produce the JSON entry as specified in the sub-skill OUTPUT FORMAT

### Step 4 — Construct JSON entry
Every entry must follow this structure:
```json
{
  "sample_id": "atomic-layer-deposition/experimental-usecase/1/figure_2",
  "_confidence": "high",
  "_chart_type": "line",
  "data_extraction": {
    "a": "```markdown\n| Col1 | Col2 |\n| --- | --- |\n| val | val |\n```"
  }
}
```

Panel labels ("a", "b", "c"...) come from visible panel labels in the figure. No visible labels → use "a" as the single key.

### Step 5 — Save incrementally
- Append each entry to results list
- After every 10 images: save to `OUTPUT_FOLDER/submission.json` and append sample_id to `OUTPUT_FOLDER/checkpoint.txt`
- After all images: do a final save

### Step 6 — Print summary
```
============================================
EXTRACTION COMPLETE
============================================
Total processed:   X
Breakdown:
  line:            X
  bar:             X
  scatter:         X
  table:           X
  heatmap:         X
  schematic:       X
  polar_radar:     X
Confidence:
  high:            X
  medium:          X
  low:             X
Errors:            X
Output:            <OUTPUT_FOLDER>/submission.json
============================================
```

---

## IMPORTANT CONSTRAINTS

- **Never skip an image** — every image must have a JSON entry, even if extraction fails
- **Fallback for errors**: if extraction fails → output `| No extractable data |\n| --- |` with `_confidence: low`
- **Do not modify originals** — read only, never write to image folder
- **Checkpoint every 10** — protects against session timeout
- **Resume mode** — if checkpoint.txt exists, skip already-processed images automatically
- **No user prompts** — do NOT ask for confirmation at any point; process all images autonomously from start to finish without pausing, confirming, or requesting approval between images or batches

---

## HOW TO USE

1. Open a new Claude Code session
2. Paste this entire SKILL.md (paths are pre-filled below)
3. Review the plan → approve
4. Wait for EXTRACTION COMPLETE summary
5. Zip `OUTPUT_FOLDER/submission.json` and upload to CodaBench

---

## USER INPUT — FILL IN BEFORE RUNNING

```
IMAGE_FOLDER:   /home/jovyan/tvinchur/sci_image_miner/ALD-E-ImageMiner/icdar2026-competition-data/test
SKILLS_FOLDER:  /home/jovyan/tvinchur/sci_image_miner/data_extraction
OUTPUT_FOLDER:  /home/jovyan/tvinchur/sci_image_miner/outputs/task2_extraction
```
