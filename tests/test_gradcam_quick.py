"""
Quick test script for the Grad-CAM endpoint.

Run with:
    pytest -v tests/test_gradcam_quick.py

Or directly:
    python tests/test_gradcam_quick.py
"""

import base64
import io

import pytest
import requests
from PIL import Image


BASE_URL = "http://localhost:8000"
SAMPLE_DIR = "static/samples"


@pytest.mark.parametrize(
    "img_filename",
    [
        "tomato_early_blight.jpg",
        "corn_common_rust.jpg",
        "potato_late_blight.jpg",
    ],
)
def test_gradcam_endpoint(img_filename: str):
    img_path = f"{SAMPLE_DIR}/{img_filename}"

    print(f"\n=== Testing: {img_filename} ===")

    with open(img_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/predict-with-gradcam",
            files={"file": (img_filename, f, "image/jpeg")},
            params={"top_k": 5},
        )

    assert resp.status_code == 200, (
        f"HTTP {resp.status_code}: {resp.text[:300]}"
    )

    data = resp.json()

    pred = data["prediction"]
    gcam = data["gradcam"]

    print(f"  Predicted class : {pred['class_name']}")
    print(f"  Confidence      : {pred['confidence']}%")
    print(f"  Grad-CAM success: {gcam['success']}")
    print(f"  Target layer    : {gcam['target_layer']}")
    print(f"  Target class idx: {gcam['target_class']}")
    print(f"  Error           : {gcam['error']}")

    assert gcam["success"], (
        f"Grad-CAM failed: {gcam['error']}"
    )

    assert gcam["target_layer"] == "top_activation"

    for key in ["original_image", "heatmap", "overlay"]:
        val = gcam.get(key, "")

        assert val.startswith(
            "data:image/jpeg;base64,"
        ), f"{key} is not a valid JPEG data URI"

        raw = base64.b64decode(val.split(",")[1])

        img = Image.open(io.BytesIO(raw))

        print(
            f"  {key:20s}: "
            f"{img.size[0]}x{img.size[1]}px  "
            f"mode={img.mode}  "
            f"size={len(raw)//1024}KB"
        )

        assert img.size[0] > 0
        assert img.size[1] > 0

    print("  PASS")


def test_original_predict_unchanged():
    print("\n=== Testing: /predict still works (backward compat) ===")

    with open(f"{SAMPLE_DIR}/healthy_apple.jpg", "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/predict",
            files={"file": ("healthy_apple.jpg", f, "image/jpeg")},
        )

    assert resp.status_code == 200, (
        f"HTTP {resp.status_code}"
    )

    data = resp.json()

    assert data["success"] is True
    assert "prediction" in data
    assert "gradcam" not in data

    print(f"  Class: {data['prediction']['class_name']}")
    print("  PASS - original /predict endpoint is unchanged")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
