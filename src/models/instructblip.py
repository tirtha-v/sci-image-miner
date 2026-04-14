"""InstructBLIP-Vicuna-7B classifier."""

import gc
import torch
from PIL import Image

from .base import VLMClassifier


class InstructBLIPClassifier(VLMClassifier):
    """Uses Salesforce/instructblip-vicuna-7b for zero-shot chart classification."""

    model_name = "instructblip"
    _MODEL_ID = "Salesforce/instructblip-vicuna-7b"

    def __init__(self, max_new_tokens: int = 50):
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

        print(f"[InstructBLIP] Loading {self._MODEL_ID} ...")
        self.processor = InstructBlipProcessor.from_pretrained(self._MODEL_ID)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            self._MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        print("[InstructBLIP] Model loaded.")

    def classify_image(self, pil_image: Image.Image, system_prompt: str, user_prompt: str) -> str:
        # InstructBLIP struggles with long prompts (325+ tokens listing all taxonomy labels).
        # Use a short open-ended prompt; postprocessor maps the answer to the taxonomy.
        short_prompt = (
            "You are a scientific figure classifier. "
            "What type of chart or figure is shown in this image? "
            "Answer with the chart type only, as briefly as possible."
        )

        inputs = self.processor(
            images=pil_image,
            text=short_prompt,
            return_tensors="pt",
        ).to(next(self.model.parameters()).device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # Trim input tokens — InstructBLIP's decoder returns full sequence
        input_len = inputs["input_ids"].shape[1]
        output = self.processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
        )[0]
        return output.strip()

    def unload_model(self) -> None:
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[InstructBLIP] Model unloaded.")
