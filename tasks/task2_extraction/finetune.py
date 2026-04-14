#!/usr/bin/env python3
"""QLoRA fine-tuning of Qwen2.5-VL-7B for data extraction (Task 2).

Reads train_extraction.csv (image_path, sample_id, panel_id, markdown_table).
Fine-tunes the model to output a markdown table from a cropped panel image.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python tasks/task2_extraction/finetune.py \
        --train-csv data/train_extraction.csv \
        --output-dir outputs/qwen2vl_extraction_qlora \
        --epochs 3 --lr 1e-4
"""

import os
import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

SYSTEM_PROMPT = (
    "You are a Vision Language Model specialized in extracting structured data "
    "from scientific figure panels into Markdown tables."
)

USER_PROMPT = (
    "Extract the quantitative data from this scientific figure panel as a Markdown table.\n"
    "- Use standard Markdown table format with | separators and a header row followed by |---|---| separator.\n"
    "- Include all data values visible in the chart (axes labels as column headers, data points as rows).\n"
    "- If no tabular data is extractable (e.g., schematic, image panel, molecular diagram), output an empty string.\n"
    "- Output ONLY the markdown table or empty string — no explanation."
)


class ExtractionSFTDataset(Dataset):
    def __init__(self, csv_path: str):
        self.samples = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                table = row["markdown_table"].strip()
                self.samples.append({
                    "image_path": row["image_path"],
                    "target": table,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s["image_path"]).convert("RGB")
        conversations = [
            {"role": "user", "content": f"<image>\n{SYSTEM_PROMPT}\n\n{USER_PROMPT}"},
            {"role": "assistant", "content": s["target"]},
        ]
        return {"image": image, "conversations": conversations}


class ExtractionCollator:
    _ASSISTANT_HEADER = "<|im_start|>assistant\n"

    def __init__(self, processor):
        self.processor = processor
        self._header_ids = processor.tokenizer.encode(
            self._ASSISTANT_HEADER, add_special_tokens=False
        )

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        full_texts = []

        for item in batch:
            convs = item["conversations"]
            user_msg = convs[0]["content"]
            if user_msg.startswith("<image>\n"):
                user_msg = user_msg[len("<image>\n"):]
            assistant_msg = convs[1]["content"]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_msg},
                    ],
                },
                {"role": "assistant", "content": assistant_msg},
            ]
            full_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            )
            full_texts.append(full_text)

        inputs = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        labels = inputs["input_ids"].clone()
        header_ids = self._header_ids
        header_len = len(header_ids)

        for i in range(labels.shape[0]):
            seq = inputs["input_ids"][i].tolist()
            mask_end = 0
            for j in range(len(seq) - header_len + 1):
                if seq[j:j + header_len] == header_ids:
                    mask_end = j + header_len
            labels[i, :mask_end] = -100

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/train_extraction.csv")
    parser.add_argument("--output-dir", default="outputs/qwen2vl_extraction_qlora")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load HF token if available
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        import huggingface_hub
        huggingface_hub.login(token=hf_token, add_to_git_credential=False)

    print(f"[Extraction QLoRA] Loading {args.model_id} in bfloat16 ...")
    processor = AutoProcessor.from_pretrained(
        args.model_id, min_pixels=64*28*28, max_pixels=256*28*28
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map="balanced",
        torch_dtype=torch.bfloat16,
    )

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = ExtractionSFTDataset(args.train_csv)
    print(f"Train: {len(train_dataset)} panels")

    collator = ExtractionCollator(processor)

    steps_per_epoch = len(train_dataset) // (args.batch_size * args.grad_accum)
    total_steps = steps_per_epoch * args.epochs

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    print("[Extraction QLoRA] Starting training...")
    trainer.train()

    adapter_path = output_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
    print(f"[Extraction QLoRA] Adapter saved to {adapter_path}")


if __name__ == "__main__":
    main()
