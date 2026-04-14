"""Phi-3.5-Vision-Instruct classifier."""

import gc
import torch
from PIL import Image

from .base import VLMClassifier


class Phi35VisionClassifier(VLMClassifier):
    """Uses microsoft/Phi-3.5-vision-instruct for zero-shot chart classification."""

    model_name = "phi_3_5_vision"
    _MODEL_ID = "microsoft/Phi-3.5-vision-instruct"

    def __init__(self, max_new_tokens: int = 50):
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        from transformers import AutoProcessor, AutoModelForCausalLM

        print(f"[Phi-3.5-Vision] Loading {self._MODEL_ID} ...")
        self.processor = AutoProcessor.from_pretrained(
            self._MODEL_ID,
            trust_remote_code=True,
            num_crops=4,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self._MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            _attn_implementation="eager",
        )
        self.model.eval()
        print("[Phi-3.5-Vision] Model loaded.")

    def classify_image(self, pil_image: Image.Image, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{system_prompt}\n\n{user_prompt}"},
        ]

        text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=text, images=[pil_image], return_tensors="pt").to(
            next(self.model.parameters()).device
        )

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=False,  # bypass DynamicCache which is incompatible with transformers 5.x
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
        print("[Phi-3.5-Vision] Model unloaded.")
