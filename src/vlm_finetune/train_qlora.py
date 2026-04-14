"""QLoRA finetuning for chart classification VLMs (LLaVA-1.6 or Llama-3.2-Vision)."""

import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import f1_score

from .dataset import ChartSFTDataset


class LLaVACollator:
    """Collate function that processes images + text for LLaVA-1.6 training."""

    def __init__(self, processor, max_length=512):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        prompts = []

        for item in batch:
            convs = item["conversations"]
            user_msg = convs[0]["content"]
            assistant_msg = convs[1]["content"]
            # LLaVA-1.6 Mistral format
            prompt = f"[INST] {user_msg} [/INST] {assistant_msg}"
            prompts.append(prompt)

        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Create labels: mask user tokens, only compute loss on assistant response
        labels = inputs["input_ids"].clone()
        for i, item in enumerate(batch):
            user_msg = item["conversations"][0]["content"]
            assistant_msg = item["conversations"][1]["content"]
            prompt = f"[INST] {user_msg} [/INST] {assistant_msg}"
            inst_end = prompt.find("[/INST]")
            if inst_end >= 0:
                prefix = prompt[:inst_end + len("[/INST] ")]
                prefix_tokens = self.processor.tokenizer(
                    prefix, return_tensors="pt", add_special_tokens=False
                )
                prefix_len = prefix_tokens["input_ids"].shape[1]
                labels[i, :prefix_len] = -100

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs


class LlamaCollator:
    """Collate function for Llama-3.2-Vision training.

    Uses apply_chat_template and finds the assistant header in token IDs to
    mask user tokens — robust to image token expansion.
    """

    # Llama 3 assistant header (tokenized separately to find mask boundary)
    _ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"

    def __init__(self, processor, max_length=768):
        self.processor = processor
        self.max_length = max_length
        self._header_ids = processor.tokenizer.encode(
            self._ASSISTANT_HEADER, add_special_tokens=False
        )

    def __call__(self, batch):
        # Resize to fit within one Mllama tile (560x560) — 4-tile default
        # causes 2+ GB SDPA allocations that OOM on 2080 Ti.
        MAX_SIZE = 560
        images = []
        for item in batch:
            img = item["image"]
            if img.width > MAX_SIZE or img.height > MAX_SIZE:
                img = img.copy()
                img.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)
            images.append(img)
        full_texts = []

        for item in batch:
            convs = item["conversations"]
            # Strip "<image>\n" prefix that ChartSFTDataset adds for LLaVA format
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
                {
                    "role": "assistant",
                    "content": assistant_msg,
                },
            ]
            full_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            )
            full_texts.append(full_text)

        # Mllama requires images as a nested list: [[img1], [img2], ...]
        inputs = self.processor(
            text=full_texts,
            images=[[img] for img in images],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Mask user tokens by finding the assistant header in each sequence
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


class Qwen2VLCollator:
    """Collate function for Qwen2.5-VL QLoRA training.

    Uses apply_chat_template and finds the assistant header token sequence
    to mask user tokens — robust to image token expansion.
    """

    _ASSISTANT_HEADER = "<|im_start|>assistant\n"

    def __init__(self, processor, max_length=768):
        self.processor = processor
        self.max_length = max_length
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
                {
                    "role": "assistant",
                    "content": assistant_msg,
                },
            ]
            full_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            )
            full_texts.append(full_text)

        # Qwen2.5-VL: no truncation — image tokens are variable-length and
        # truncation causes a token-count mismatch validation error.
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


def train_qlora(
    train_csv: str = "data/train_panels.csv",
    dev_csv: str = "data/dev_panels.csv",
    output_dir: str = "outputs/vlm_finetune/llama_qlora",
    model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_target_modules: list = None,
    device: str = "cuda:0",
    resume_from: str = None,
):
    """Finetune a vision-language model with QLoRA.

    Supports:
        - meta-llama/Llama-3.2-11B-Vision-Instruct  (MllamaForConditionalGeneration)
        - Qwen/Qwen2.5-VL-7B-Instruct               (Qwen2VLForConditionalGeneration)
        - llava-hf/llava-v1.6-mistral-7b-hf          (LlavaNextForConditionalGeneration)
    """
    from ..prompts import SYSTEM_PROMPT, build_classification_prompt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, user_prompt = build_classification_prompt(selective_descriptions=True)

    is_llama = "llama" in model_id.lower()
    is_qwen = "qwen" in model_id.lower()

    print(f"[QLoRA] Loading {model_id} in 4-bit on {device} ...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    if is_qwen:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        # Qwen2.5-VL-7B in bfloat16 is ~14GB — too large for a single RTX 2080 Ti.
        # Use device_map="balanced" to spread across all visible GPUs.
        # Set CUDA_VISIBLE_DEVICES externally (e.g. 0,1 for 2 GPUs = 21GB).
        # Limit image resolution to cap visual token count.
        # Default max_pixels (~1M) causes 7+ GB SDPA allocations on 2080 Ti.
        # 256*28*28 = ~200K pixels → ≤1024 visual tokens per image.
        processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=64*28*28, max_pixels=256*28*28
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="balanced",
            torch_dtype=torch.bfloat16,
        )
        collator = Qwen2VLCollator(processor)
    elif is_llama:
        from transformers import AutoProcessor, MllamaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_id)
        # Use device_map="auto" to spread across all visible GPUs.
        # Set CUDA_VISIBLE_DEVICES externally to control which GPUs are used.
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="balanced",
            torch_dtype=torch.bfloat16,
        )
        collator = LlamaCollator(processor)
    else:
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

        processor = LlavaNextProcessor.from_pretrained(model_id)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
        )
        collator = LLaVACollator(processor)

    if is_qwen:
        # Qwen is loaded in bfloat16 (not 4-bit) — prepare_model_for_kbit_training
        # would cast 1D params to float32, blowing past GPU memory limits.
        # Instead, just freeze the base model and enable gradient checkpointing.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    else:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        # use_reentrant=False avoids the "backward twice" error with Mllama + PEFT
        model.enable_input_require_grads()

    if resume_from:
        # Load existing adapter weights (safetensors) — fresh optimizer state
        from peft import PeftModel
        print(f"[QLoRA] Loading adapter from {resume_from} ...")
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
    else:
        target_modules = lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = ChartSFTDataset(train_csv, SYSTEM_PROMPT, user_prompt)
    dev_dataset = ChartSFTDataset(dev_csv, SYSTEM_PROMPT, user_prompt)
    print(f"Train: {len(train_dataset)}, Dev: {len(dev_dataset)}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
    )

    print("[QLoRA] Starting training...")
    trainer.train()

    adapter_path = output_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
    print(f"[QLoRA] Adapter saved to {adapter_path}")

    return str(adapter_path)
