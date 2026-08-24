from __future__ import annotations

from functools import lru_cache

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"

LABELS = [
    "a photograph of a plant leaf",
    "a photograph of a non-plant object",
]


@lru_cache(maxsize=1)
def _load_model():
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)

    model.eval()

    return processor, model


def is_plant_image(
    image: Image.Image,
    threshold: float = 0.65,
) -> tuple[bool, float]:
    """
    Determine whether an image is more likely to be a plant leaf.

    Returns
    -------
    (is_plant, confidence)
    """

    processor, model = _load_model()

    image = image.convert("RGB")

    inputs = processor(
        text=LABELS,
        images=image,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = outputs.logits_per_image.softmax(dim=1)[0]

    plant_probability = float(probabilities[0])

    return (
        plant_probability >= threshold,
        plant_probability,
    )