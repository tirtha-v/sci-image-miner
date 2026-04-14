"""Dataset for VLM supervised finetuning on chart classification."""

import csv
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class ChartSFTDataset(Dataset):
    """Converts panel crops + labels into conversation format for SFT.

    Each sample is a dict with:
        - image: PIL Image
        - conversations: list of {role, content} dicts
    """

    def __init__(self, csv_path: str, system_prompt: str, user_prompt_template: str):
        """
        Args:
            csv_path: Path to panels CSV (columns: image_path, label, sample_id, panel_id)
            system_prompt: System prompt text
            user_prompt_template: User prompt with {taxonomy} placeholder (already formatted)
        """
        self.samples = []
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt_template

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["label"].strip():
                    self.samples.append({
                        "image_path": row["image_path"],
                        "label": row["label"].strip(),
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        conversations = [
            {
                "role": "user",
                "content": f"<image>\n{self.system_prompt}\n\n{self.user_prompt}",
            },
            {
                "role": "assistant",
                "content": sample["label"],
            },
        ]

        return {
            "image": image,
            "conversations": conversations,
            "label": sample["label"],
        }


def create_sft_jsonl(csv_path: str, output_path: str, system_prompt: str, user_prompt: str):
    """Convert CSV to JSONL format for training frameworks that expect it.

    Each line: {"image": path, "conversations": [...]}
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path) as f_in, open(output_path, "w") as f_out:
        reader = csv.DictReader(f_in)
        count = 0
        for row in reader:
            if not row["label"].strip():
                continue
            entry = {
                "image": row["image_path"],
                "conversations": [
                    {
                        "role": "user",
                        "content": f"<image>\n{system_prompt}\n\n{user_prompt}",
                    },
                    {
                        "role": "assistant",
                        "content": row["label"].strip(),
                    },
                ],
            }
            f_out.write(json.dumps(entry) + "\n")
            count += 1

    print(f"[sft_dataset] Wrote {count} samples to {output_path}")
