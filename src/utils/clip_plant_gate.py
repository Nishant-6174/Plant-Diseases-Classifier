"""
clip_plant_gate.py
------------------

Zero-shot plant/non-plant image gate using OpenAI CLIP.

This module is used BEFORE the disease classifier to prevent
arbitrary non-plant images from reaching the EfficientNet model.
"""

from functools import lru_cache
from typing import Tuple

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "openai/clip-vit-base-patch32"

# CLIP plant confidence threshold.
#
# Values observed during testing:
# Leaf      -> ~0.98
# Dustbin   -> ~0.16
# Form      -> ~0.15
# Car       -> ~0.30
#
# 0.70 provides a conservative separation between
# plant and obvious non-plant images.
PLANT_THRESHOLD = 0.40


# ---------------------------------------------------------
# Load CLIP only once
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def _load_clip():
    """
    Load the CLIP model and processor once.

    Hugging Face automatically caches the model locally
    after the first download.
    """

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)

    model.eval()

    return processor, model


# ---------------------------------------------------------
# Plant probability
# ---------------------------------------------------------

def get_plant_confidence(image: Image.Image) -> float:
    """
    Estimate whether an image is a plant/leaf using CLIP.

    Uses paired positive/negative prompts so the score
    represents plant-vs-non-plant similarity rather than
    competition among many unrelated classes.

    Returns
    -------
    float
        Value between 0.0 and 1.0.
    """

    processor, model = _load_clip()

    image = image.convert("RGB")

    positive_prompts = [
        "a photo of a plant leaf",
        "a close-up photo of a plant leaf",
        "a photo of a crop leaf",
        "a photo of foliage",
        "a photo of a diseased plant leaf",
    ]

    negative_prompts = [
        "a photo of a non-plant object",
        "a photo of an object",
        "a photo of a vehicle",
        "a photo of an animal",
        "a photo of a person",
        "a photo of a building",
    ]

    plant_scores = []
    non_plant_scores = []

    with torch.no_grad():

        for positive, negative in zip(
            positive_prompts[:3],
            negative_prompts[:3],
        ):

            inputs = processor(
                text=[positive, negative],
                images=image,
                return_tensors="pt",
                padding=True,
            )

            outputs = model(**inputs)

            logits = outputs.logits_per_image[0]

            probabilities = torch.softmax(logits, dim=0)

            plant_scores.append(
                probabilities[0].item()
            )

            non_plant_scores.append(
                probabilities[1].item()
            )

    plant_score = sum(plant_scores) / len(plant_scores)

    return float(plant_score)
# ---------------------------------------------------------
# Main gate
# ---------------------------------------------------------

def is_plant_image(
    image: Image.Image,
) -> Tuple[bool, float]:
    """
    Determine whether an image should be allowed into the
    plant disease classifier.

    Returns
    -------
    Tuple[bool, float]
        (is_plant, plant_confidence)
    """

    confidence = get_plant_confidence(image)

    is_plant = confidence >= PLANT_THRESHOLD

    return is_plant, confidence


# ---------------------------------------------------------
# Human-readable validation
# ---------------------------------------------------------

def validate_plant_image(
    image: Image.Image,
) -> Tuple[bool, float]:
    """
    Validate an uploaded image using CLIP.

    Returns
    -------
    Tuple[bool, float]
        (is_plant, plant_confidence)

    Examples
    --------
    (True, 0.8956)
    (False, 0.3018)
    """

    try:
        return is_plant_image(image)

    except Exception as exc:
        raise RuntimeError(
            f"Plant image validation failed: {exc}"
        ) from exc