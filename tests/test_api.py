import io
import json
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.config import settings

client = TestClient(app)

def create_dummy_image(color=(255, 0, 0), size=(100, 100)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "device" in data
    assert "cuda_available" in data
    assert "dtype" in data

def test_query_endpoint_mock():
    # Force mock model for deterministic API test
    settings.use_mock_model = True
    
    metadata = {
        "question": "Where is the chair?",
        "query_scope": "current_view",
        "views": [{"view_id": "current", "role": "context"}]
    }
    
    dummy_img = create_dummy_image()
    files = [
        ("images", ("current.jpg", dummy_img, "image/jpeg"))
    ]
    data = {
        "metadata": json.dumps(metadata)
    }
    
    res = client.post("/api/v1/vision/query", data=data, files=files)
    assert res.status_code == 200
    json_resp = res.json()
    assert "chair" in json_resp["answer"].lower()
    assert len(json_resp["observations"]) >= 1
    assert json_resp["observations"][0]["label"] == "armchair"
    assert json_resp["metrics"] is not None
    assert json_resp["metrics"]["total_ms"] > 0
