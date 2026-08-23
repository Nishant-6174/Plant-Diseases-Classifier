"""
test_validation.py
------------------
Tests for the conservative image-validation heuristic and its integration
into the /predict and /predict-with-gradcam API endpoints.

Uses FastAPI TestClient — the same pattern as test_api.py — so no live
server is needed and the model is loaded only once per pytest session.

Unit tests cover validate_image() directly with PIL images.
Integration tests exercise the HTTP endpoints via TestClient.

Positive cases use real sample images from static/samples/ OR synthetic
green/yellow images that look plant-like.
Negative cases use synthetically generated solid non-organic colour blocks.

NOTE: These tests do NOT modify or remove any of the 15 existing tests.
"""
import io
import os

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app import app
from src.utils.image_validator import validate_image

# Shared client — same app singleton, model loaded once per process.
client = TestClient(app)

SAMPLES_DIR = os.path.join("static", "samples")


# ===========================================================================
# Helpers
# ===========================================================================

def _make_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _post_predict(img: Image.Image, top_k: int = 3):
    return client.post(
        "/predict",
        files={"file": ("test.jpg", _make_jpeg_bytes(img), "image/jpeg")},
        params={"top_k": top_k},
    )


def _post_gradcam(img: Image.Image, top_k: int = 3):
    return client.post(
        "/predict-with-gradcam",
        files={"file": ("test.jpg", _make_jpeg_bytes(img), "image/jpeg")},
        params={"top_k": top_k},
    )


# ===========================================================================
# Unit tests: validate_image() function
# No HTTP requests — just PIL → validator logic.
# ===========================================================================

class TestValidateImageUnit:
    """Direct unit tests for the validate_image() heuristic."""

    # ── Positive: real sample leaf images ───────────────────────────────────

    @pytest.mark.parametrize("filename", [
        "tomato_early_blight.jpg",
        "corn_common_rust.jpg",
        "potato_late_blight.jpg",
        "healthy_apple.jpg",
    ])
    def test_real_leaf_samples_pass(self, filename):
        """All real leaf samples must pass the conservative validator."""
        path = os.path.join(SAMPLES_DIR, filename)
        if not os.path.exists(path):
            pytest.skip(f"Sample not found: {path}")
        img = Image.open(path).convert("RGB")
        is_valid, reason = validate_image(img)
        assert is_valid, (
            f"Validator incorrectly rejected '{filename}'. Reason: {reason}"
        )

    def test_synthetic_green_passes(self):
        """
        Synthetic solid-green image (color=(50,150,50)) used by the existing
        test_predict_multipart_file test must pass the validator.
        The green hue falls squarely in the plant-colour range.
        """
        img = Image.new("RGB", (224, 224), color=(50, 150, 50))
        is_valid, _ = validate_image(img)
        assert is_valid, "Solid green image should pass — organic colour present."

    def test_synthetic_yellow_passes(self):
        """Yellow/brown tones typical of diseased leaves should pass."""
        img = Image.new("RGB", (224, 224), color=(180, 140, 30))
        is_valid, _ = validate_image(img)
        assert is_valid, "Yellow/brown image should pass (diseased leaf tones)."

    # ── Negative: obviously non-organic solid-colour images ─────────────────

    def test_solid_blue_rejected(self):
        """Pure blue: zero organic colour, zero texture, zero edges → reject."""
        img = Image.new("RGB", (224, 224), color=(0, 0, 255))
        is_valid, reason = validate_image(img)
        assert not is_valid, "Solid blue should be rejected."

    def test_solid_gray_rejected(self):
        """Pure neutral gray: zero organic colour, zero texture → reject."""
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        is_valid, reason = validate_image(img)
        assert not is_valid, "Solid gray should be rejected."

    def test_solid_black_rejected(self):
        """Solid black image must be rejected."""
        img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        is_valid, reason = validate_image(img)
        assert not is_valid, "Solid black should be rejected."

    def test_solid_white_rejected(self):
        """Solid white image must be rejected."""
        img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        is_valid, reason = validate_image(img)
        assert not is_valid, "Solid white should be rejected."

    def test_rejected_returns_nonempty_reason(self):
        """A rejected image must return a non-empty reason string."""
        img = Image.new("RGB", (224, 224), color=(0, 0, 255))
        is_valid, reason = validate_image(img)
        assert not is_valid
        assert isinstance(reason, str) and len(reason) > 0

    def test_valid_returns_ok(self):
        """A passing image must return reason == 'OK'."""
        img = Image.new("RGB", (224, 224), color=(50, 150, 50))
        is_valid, reason = validate_image(img)
        assert is_valid
        assert reason == "OK"


# ===========================================================================
# Integration tests: /predict endpoint (TestClient)
# ===========================================================================

class TestPredictValidation:
    """Endpoint-level tests for validation in /predict."""

    def test_green_leaf_image_returns_200(self):
        """
        A green synthetic image passes validation and reaches the classifier.
        Verifies that the validator does NOT block valid plant-like inputs.
        """
        img = Image.new("RGB", (224, 224), color=(50, 150, 50))
        resp = _post_predict(img)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "prediction" in data

    def test_real_leaf_from_disk_returns_200(self):
        """A real sample leaf image must reach the classifier (HTTP 200)."""
        path = os.path.join(SAMPLES_DIR, "tomato_early_blight.jpg")
        if not os.path.exists(path):
            pytest.skip("Sample not found.")
        with open(path, "rb") as f:
            resp = client.post(
                "/predict",
                files={"file": ("tomato_early_blight.jpg", f, "image/jpeg")},
                params={"top_k": 3},
            )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_solid_blue_returns_400(self):
        """/predict must return HTTP 400 for a solid blue image."""
        img = Image.new("RGB", (224, 224), color=(0, 0, 255))
        resp = _post_predict(img)
        assert resp.status_code == 400
        data = resp.json()
        assert data["success"] is False
        assert data["error_type"] == "invalid_image"
        assert "message" in data

    def test_solid_gray_returns_400(self):
        """/predict must return HTTP 400 for a solid gray image."""
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        resp = _post_predict(img)
        assert resp.status_code == 400
        data = resp.json()
        assert data["success"] is False
        assert data["error_type"] == "invalid_image"

    def test_invalid_response_schema(self):
        """HTTP 400 body must contain success, error_type, and message fields."""
        img = Image.new("RGB", (224, 224), color=(0, 0, 255))
        resp = _post_predict(img)
        assert resp.status_code == 400
        data = resp.json()
        for key in ("success", "error_type", "message"):
            assert key in data, f"Missing field: {key}"
        assert data["success"] is False
        assert data["error_type"] == "invalid_image"


# ===========================================================================
# Integration tests: /predict-with-gradcam endpoint (TestClient)
# ===========================================================================

class TestGradcamValidation:
    """Endpoint-level tests for validation in /predict-with-gradcam."""

    def test_green_leaf_gradcam_returns_200(self):
        """A green synthetic image must pass validation and return gradcam data."""
        img = Image.new("RGB", (224, 224), color=(50, 150, 50))
        resp = _post_gradcam(img)
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "gradcam" in data

    def test_solid_blue_gradcam_returns_400_no_gradcam(self):
        """
        /predict-with-gradcam must return HTTP 400 for a solid blue image.
        Critically: the 'gradcam' key must be ABSENT (classifier not run).
        """
        img = Image.new("RGB", (224, 224), color=(0, 0, 255))
        resp = _post_gradcam(img)
        assert resp.status_code == 400
        data = resp.json()
        assert data["success"] is False
        assert data["error_type"] == "invalid_image"
        assert "gradcam" not in data, (
            "gradcam key must not exist when validation rejects the image"
        )

    def test_solid_gray_gradcam_returns_400(self):
        """/predict-with-gradcam must return HTTP 400 for a solid gray image."""
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        resp = _post_gradcam(img)
        assert resp.status_code == 400
        data = resp.json()
        assert data["success"] is False
        assert data["error_type"] == "invalid_image"
