"""CNN model definitions via timm."""

import timm
import torch.nn as nn


def create_model(model_name: str, num_classes: int, pretrained: bool = True, img_size: int = None) -> nn.Module:
    """Create a timm model for classification.

    Args:
        model_name: timm model name (e.g. 'efficientnet_b0', 'inception_resnet_v2')
        num_classes: Number of output classes
        pretrained: Whether to load ImageNet pretrained weights
        img_size: Override input image size (resizes position embeddings for ViT models)

    Returns:
        nn.Module with the classifier head replaced
    """
    kwargs = {"pretrained": pretrained, "num_classes": num_classes}
    if img_size is not None:
        kwargs["img_size"] = img_size
    model = timm.create_model(model_name, **kwargs)
    return model


def freeze_backbone(model: nn.Module):
    """Freeze all layers except the classifier head."""
    # timm models use various names: classifier, classif, fc, head
    head_keywords = ("classifier", "classif", "fc", "head")
    for name, param in model.named_parameters():
        if not any(kw in name for kw in head_keywords):
            param.requires_grad = False


def unfreeze_all(model: nn.Module):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


# Standard model configs.
# img_size: used for dataset transforms (resize target).
# timm_img_size: if set, also passed to timm.create_model() as img_size kwarg
#   (needed for ViT/attention models to resize positional embeddings).
#   CNN models like InceptionResNetV2 and EfficientNet do NOT accept this kwarg.
MODEL_CONFIGS = {
    "efficientnet_b0": {
        "timm_name": "efficientnet_b0",
        "img_size": 224,
    },
    "inception_resnet_v2": {
        "timm_name": "inception_resnet_v2",
        "img_size": 299,
        # no timm_img_size — InceptionResNetV2 does not accept img_size kwarg
    },
    "swinv2_base": {
        "timm_name": "swinv2_base_window16_256",
        "img_size": 256,
        "timm_img_size": 256,
    },
    "convnextv2_base": {
        "timm_name": "convnextv2_base.fcmae_ft_in22k_in1k",
        "img_size": 224,
    },
    "vit_base": {
        "timm_name": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        "img_size": 224,
        "timm_img_size": 224,
    },
}
