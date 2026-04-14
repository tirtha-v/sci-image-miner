#!/usr/bin/env python3
"""QLoRA fine-tuning of Qwen2.5-VL-7B for VQA (Task 4).

Reads data/train_vqa.csv (image_path, sample_id, panel_id,
question_type, question, answer_type, answer).
Fine-tunes the model to answer VQA questions about scientific figure panels.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python run_qwen_vqa_finetune.py \
        --train-csv data/train_vqa.csv \
        --output-dir outputs/qwen2vl_vqa_qlora \
        --epochs 3 --lr 1e-4
"""

import argparse
import csv
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
    "You are an expert in materials science specializing in atomic layer deposition (ALD) "
    "and atomic layer etching (ALE). Answer questions about scientific figure panels accurately."
)


def build_user_prompt(question: str, answer_type: str) -> str:
    hints = {
        "Yes/No": "Answer with 'Yes' or 'No' followed by a brief explanation.",
        "Factoid": "Provide a concise factual answer.",
        "List": "Provide a bulleted or comma-separated list.",
        "Paragraph": "Provide a detailed 2-4 sentence answer with specific values where visible.",
    }
    hint = hints.get(answer_type, "")
    return f"Question: {question}\n{hint}"


class VQASFTDataset(Dataset):
    def __init__(self, csv_path: str):
        self.samples = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                answer = row["answer"].strip()
                question = row["question"].strip()
                if answer and question:
                    self.samples.append({
                        "image_path": row["image_path"],
                        "question": question,
                        "answer_type": row["answer_type"],
                        "answer": answer,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s["image_path"]).convert("RGB")
        user_prompt = build_user_prompt(s["question"], s["answer_type"])
        conversations = [
            {"role": "user", "content": f"<image>\n{SYSTEM_PROMPT}\n\n{user_prompt}"},
            {"role": "assistant", "content": s["answer"]},
        ]
        return {"image": image, "conversations": conversations}


class VQACollator:
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
    parser.add_argument("--train-csv", default="data/train_vqa.csv")
    parser.add_argument("--output-dir", default="outputs/qwen2vl_vqa_qlora")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for token_path in [
        "/home/jovyan/tvinchur/hf_token_for_cluster.txt",
        "/home/jovyan/shared-scratch-tvinchur-pvc/tvinchur/hf_token_for_cluster.txt",
    ]:
        if Path(token_path).exists():
            import huggingface_hub
            huggingface_hub.login(token=Path(token_path).read_text().strip(), add_to_git_credential=False)
            break

    print(f"[VQA QLoRA] Loading {args.model_id} ...")
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

    train_dataset = VQASFTDataset(args.train_csv)
    print(f"[VQA QLoRA] Train: {len(train_dataset)} QA pairs")

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
        data_collator=VQACollator(processor),
    )

    print("[VQA QLoRA] Starting training...")
    trainer.train()

    adapter_path = output_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
    print(f"[VQA QLoRA] Adapter saved to {adapter_path}")


if __name__ == "__main__":
    main()
