"""LLaVA-1.6-Mistral-7B classifier."""

import gc
import torch
from PIL import Image

from .base import VLMClassifier


class LLaVAClassifier(VLMClassifier):
    """Uses llava-hf/llava-v1.6-mistral-7b-hf for zero-shot chart classification."""

    model_name = "llava"
    _MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

    def __init__(self, max_new_tokens: int = 50):
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

        print(f"[LLaVA] Loading {self._MODEL_ID} ...")
        self.processor = LlavaNextProcessor.from_pretrained(self._MODEL_ID)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self._MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        print("[LLaVA] Model loaded.")

    def classify_image(self, pil_image: Image.Image, system_prompt: str, user_prompt: str) -> str:
        # LLaVA-1.6 Mistral uses [INST] format
        full_prompt = (
            f"[INST] <image>\n{system_prompt}\n\n{user_prompt} [/INST]"
        )

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
        print("[LLaVA] Model unloaded.")
