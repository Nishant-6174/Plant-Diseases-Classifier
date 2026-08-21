import base64
import io
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple
from PIL import Image

from src.exception import PlantDiseaseException
from src.logger import logger


def read_json(path_to_json: str) -> Dict[str, Any]:
    """
    Reads a JSON file safely and returns its contents.
    """
    try:
        with open(path_to_json, "r", encoding="utf-8") as f:
            content = json.load(f)
        logger.info(f"JSON file loaded successfully from: {path_to_json}")
        return content
    except Exception as e:
        logger.error(f"Error loading JSON file from {path_to_json}: {e}")
        raise PlantDiseaseException(e, sys)


def save_json(path_to_json: str, data: Any) -> None:
    """
    Saves python dictionary or list to a JSON file.
    """
    try:
        os.makedirs(os.path.dirname(path_to_json), exist_ok=True)
        with open(path_to_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"JSON file saved successfully at: {path_to_json}")
    except Exception as e:
        logger.error(f"Error saving JSON file at {path_to_json}: {e}")
        raise PlantDiseaseException(e, sys)


def decode_base64_image(base64_string: str) -> Image.Image:
    """
    Decodes a base64 string into a PIL Image.
    """
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        raise PlantDiseaseException(f"Invalid base64 image string: {e}", sys)


def encode_image_to_base64(image_path: str) -> str:
    """
    Encodes an image file on disk into a base64 string.
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception as e:
        logger.error(f"Failed to encode image to base64: {e}")
        raise PlantDiseaseException(e, sys)


def parse_class_metadata(class_name: str) -> Dict[str, Any]:
    """
    Parses a raw class name into readable plant name, condition, and health status.
    Examples:
        'healthy_apple' -> plant: 'Apple', condition: 'Healthy', is_healthy: True
        'apple_apple_scab' -> plant: 'Apple', condition: 'Apple Scab', is_healthy: False
        'bell_pepper_bacterial_spot' -> plant: 'Bell Pepper', condition: 'Bacterial Spot', is_healthy: False
        'diseased_rice' -> plant: 'Rice', condition: 'Diseased', is_healthy: False
    """
    raw = class_name.strip()
    is_healthy = raw.startswith("healthy_") or "_healthy" in raw
    
    if raw.startswith("healthy_"):
        plant = raw.replace("healthy_", "").replace("_", " ").title()
        condition = "Healthy"
    elif raw.startswith("diseased_"):
        plant = raw.replace("diseased_", "").replace("_", " ").title()
        condition = "General Disease / Leaf Infection"
    else:
        # Check known multi-word plants: 'bell_pepper', 'groundnut'
        if raw.startswith("bell_pepper_"):
            plant = "Bell Pepper"
            condition = raw.replace("bell_pepper_", "").replace("_", " ").title()
        elif raw.startswith("groundnut_"):
            plant = "Groundnut (Peanut)"
            condition = raw.replace("groundnut_", "").replace("_", " ").title()
        else:
            parts = raw.split("_")
            plant = parts[0].title()
            condition = " ".join(parts[1:]).title()
            # Clean duplicate plant name if present e.g. 'apple_apple_scab'
            if condition.lower().startswith(plant.lower() + " "):
                condition = condition[len(plant) + 1:]

    display_name = f"{plant} - {condition}" if condition != "Healthy" else f"{plant} (Healthy)"

    return {
        "raw_class": raw,
        "plant": plant,
        "condition": condition,
        "display_name": display_name,
        "is_healthy": is_healthy
    }
