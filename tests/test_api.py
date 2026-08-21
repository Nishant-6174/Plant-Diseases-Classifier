import io
import base64
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app import app

client = TestClient(app)


def test_health_endpoint():
    """
    Test GET /health endpoint returns HTTP 200 and healthy status.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["total_classes"] == 86
    assert data["service"] == "Plant Disease Classifier API"


def test_get_classes_endpoint():
    """
    Test GET /api/classes returns all 86 registered classes.
    """
    response = client.get("/api/classes")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 86
    assert len(data["classes"]) == 86


def test_get_class_info_endpoint():
    """
    Test GET /api/class-info for a known class.
    """
    response = client.get("/api/class-info/tomato_early_blight")
    assert response.status_code == 200
    data = response.json()
    assert data["plant"] == "Tomato"
    assert "symptoms" in data
    assert "organic_treatment" in data


def test_predict_multipart_file():
    """
    Test POST /predict with an uploaded multipart JPEG file.
    """
    # Create in-memory dummy image
    img = Image.new("RGB", (224, 224), color=(50, 150, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("test_leaf.jpg", buf, "image/jpeg")}
    response = client.post("/predict", files=files, params={"top_k": 3})

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "prediction" in data
    assert len(data["top_k_predictions"]) == 3


def test_predict_base64_endpoint():
    """
    Test POST /predict-base64 with a valid base64 image string.
    """
    img = Image.new("RGB", (224, 224), color=(70, 180, 70))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{encoded}"

    payload = {"image": data_uri, "top_k": 3}
    response = client.post("/predict-base64", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "prediction" in data
