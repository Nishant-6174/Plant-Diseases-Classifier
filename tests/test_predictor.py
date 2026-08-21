import numpy as np
from PIL import Image
import pytest

from src.components.predictor import PlantDiseasePredictor
from src.utils.common import parse_class_metadata


def test_parse_class_metadata():
    """
    Test metadata parser for healthy and diseased strings.
    """
    m1 = parse_class_metadata("tomato_early_blight")
    assert m1["plant"] == "Tomato"
    assert m1["condition"] == "Early Blight"
    assert m1["is_healthy"] is False

    m2 = parse_class_metadata("healthy_apple")
    assert m2["plant"] == "Apple"
    assert m2["condition"] == "Healthy"
    assert m2["is_healthy"] is True

    m3 = parse_class_metadata("bell_pepper_bacterial_spot")
    assert m3["plant"] == "Bell Pepper"
    assert m3["is_healthy"] is False


def test_image_preprocessing():
    """
    Test that preprocessing correctly resizes and shapes image to (1, 224, 224, 3).
    """
    predictor = PlantDiseasePredictor()
    dummy_img = Image.new("RGB", (500, 350), color=(100, 200, 100))
    tensor = predictor.preprocess_image(dummy_img)

    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32


def test_prediction_output_structure():
    """
    Test that prediction returns valid dictionary with Top-1 and Top-K results.
    """
    predictor = PlantDiseasePredictor()
    dummy_img = Image.new("RGB", (224, 224), color=(60, 140, 60))
    res = predictor.predict(dummy_img, top_k=5)

    assert res["success"] is True
    assert "prediction" in res
    assert "top_k_predictions" in res
    assert len(res["top_k_predictions"]) == 5

    pred = res["prediction"]
    assert "plant" in pred
    assert "condition" in pred
    assert "confidence" in pred
    assert "symptoms" in pred
    assert "organic_treatment" in pred
    assert "chemical_treatment" in pred
    assert "prevention" in pred
