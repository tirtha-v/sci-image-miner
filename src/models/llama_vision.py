"""Llama-3.2-11B-Vision-Instruct classifier."""

import gc
import torch
from PIL import Image

from .base import VLMClassifier


class LlamaVisionClassifier(VLMClassifier):
    """Uses meta-llama/Llama-3.2-11B-Vision-Instruct for zero-shot chart classification."""

    model_name = "llama_vision"
    _MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    def __init__(self, max_new_tokens: int = 50):
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        from transformers import AutoProcessor, MllamaForConditionalGeneration

        print(f"[LlamaVision] Loading {self._MODEL_ID} ...")
        self.processor = AutoProcessor.from_pretrained(self._MODEL_ID)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self._MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        print("[LlamaVision] Model loaded.")

    def classify_image(self, pil_image: Image.Image, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=prompt,
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
        print("[LlamaVision] Model unloaded.")
