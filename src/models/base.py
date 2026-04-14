"""Abstract base class for VLM classifiers."""

from abc import ABC, abstractmethod
from PIL import Image


class VLMClassifier(ABC):
    """Abstract interface for vision-language model classifiers."""

    model_name: str = "base"

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights to GPU."""

    @abstractmethod
    def classify_image(self, pil_image: Image.Image, system_prompt: str, user_prompt: str) -> str:
        """Run inference on a single panel image.

        Args:
            pil_image: Cropped panel image (RGB PIL Image)
            system_prompt: System message for the model
            user_prompt: User message containing taxonomy and instruction

        Returns:
            Raw model text output (to be post-processed)
        """

    @abstractmethod
    def unload_model(self) -> None:
        """Free GPU memory."""
