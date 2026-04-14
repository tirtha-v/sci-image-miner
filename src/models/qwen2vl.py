"""Qwen2.5-VL-7B-Instruct classifier."""

import gc
import torch
from PIL import Image

from .base import VLMClassifier


class Qwen2VLClassifier(VLMClassifier):
    """Uses Qwen/Qwen2.5-VL-7B-Instruct for zero-shot chart classification."""

    model_name = "qwen2vl"
    _MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

    def __init__(self, max_new_tokens: int = 50, load_in_4bit: bool = False):
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig

        print(f"[Qwen2VL] Loading {self._MODEL_ID} (4bit={self.load_in_4bit}) ...")
        self.processor = AutoProcessor.from_pretrained(self._MODEL_ID)
        load_kwargs = {"device_map": "auto"}
        if self.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self._MODEL_ID, **load_kwargs
        )
        self.model.eval()
        print("[Qwen2VL] Model loaded.")

    def classify_image(self, pil_image: Image.Image, system_prompt: str, user_prompt: str) -> str:
        from qwen_vl_utils import process_vision_info

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(next(self.model.parameters()).device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # Trim the input tokens from the output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip()

    def unload_model(self) -> None:
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Qwen2VL] Model unloaded.")
