"""Inference with QLoRA-finetuned LLaVA model."""

import gc
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

from ..models.base import VLMClassifier


class FineunedLLaVAClassifier(VLMClassifier):
    """LLaVA-1.6 with QLoRA adapter for inference."""

    model_name = "llava_finetuned"

    def __init__(
        self,
        base_model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        adapter_path: str = "outputs/vlm_finetune/llava_qlora/adapter",
        max_new_tokens: int = 50,
    ):
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        print(f"[FT-LLaVA] Loading base model {self.base_model_id} ...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.processor = LlavaNextProcessor.from_pretrained(self.base_model_id)
        base_model = LlavaNextForConditionalGeneration.from_pretrained(
            self.base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        print(f"[FT-LLaVA] Loading adapter from {self.adapter_path} ...")
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self.model.eval()
        print("[FT-LLaVA] Model loaded.")

    def classify_image(self, pil_image: Image.Image, system_prompt: str, user_prompt: str) -> str:
        full_prompt = f"[INST] <image>\n{system_prompt}\n\n{user_prompt} [/INST]"

        inputs = self.processor(
            text=full_prompt,
            images=pil_image,
            return_tensors="pt",
        ).to(next(self.model.parameters()).device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        output = self.processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return output.strip()

    def unload_model(self) -> None:
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[FT-LLaVA] Model unloaded.")
