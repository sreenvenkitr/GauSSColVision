import io
import json
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.config import settings
from app.schemas.vision import VisionQueryRequest, IntentType
from app.services.sam2_service import get_sam2_service
from app.services.visual_reasoning import VisualReasoningService

client = TestClient(app)

def create_test_image(color=(120, 80, 40), size=(200, 200)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def test_sam2_service_initialization():
    service = get_sam2_service()
    assert service.enabled is True
    # Test model loading on CPU
    loaded = service.load_model()
    assert loaded is True
    assert service.ready is True
    info = service.get_info()
    assert info["model"] == "sam2.1_hiera_tiny"
    assert info["ready"] is True

def test_sam_segment_endpoint():
    img_bytes = create_test_image()
    files = {
        "image": ("test.jpg", img_bytes, "image/jpeg")
    }
    data = {
        "x_min": 0.2,
        "y_min": 0.2,
        "x_max": 0.8,
        "y_max": 0.8
    }
    res = client.post("/api/v1/sam/segment", data=data, files=files)
    assert res.status_code == 200
    json_data = res.json()
    assert json_data["status"] == "ok"
    assert json_data["confidence"] > 0
    assert json_data["width"] == 200
    assert json_data["height"] == 200
    assert json_data["foreground_pixel_count"] > 0
    assert json_data["mask_png_base64"].startswith("data:image/png;base64,")

def test_visual_query_does_not_invoke_sam(monkeypatch):
    settings.use_mock_model = True
    sam_service = get_sam2_service()
    
    sam_invoked = False
    original_segment = sam_service.segment_image
    def mock_segment(*args, **kwargs):
        nonlocal sam_invoked
        sam_invoked = True
        return original_segment(*args, **kwargs)
    
    monkeypatch.setattr(sam_service, "segment_image", mock_segment)
    
    metadata = {
        "question": "What color is the desk?",
        "query_scope": "current_view",
        "views": [{"view_id": "current", "role": "context"}]
    }
    dummy_img = create_test_image()
    files = [("images", ("current.jpg", dummy_img, "image/jpeg"))]
    data = {"metadata": json.dumps(metadata)}
    
    res = client.post("/api/v1/vision/query", data=data, files=files)
    assert res.status_code == 200
    resp_json = res.json()
    assert resp_json["intent"] == "VISUAL_QUERY"
    assert resp_json["requires_segmentation"] is False
    assert resp_json["segmentation"] is None
    assert sam_invoked is False

def test_select_object_invokes_sam():
    settings.use_mock_model = True
    metadata = {
        "question": "Select the brown chair.",
        "query_scope": "scene",
        "views": [
            {"view_id": "current", "role": "scene_view"},
            {"view_id": "yaw_plus_45", "role": "scene_view"}
        ]
    }
    dummy_img1 = create_test_image()
    dummy_img2 = create_test_image()
    files = [
        ("images", ("current.jpg", dummy_img1, "image/jpeg")),
        ("images", ("yaw_plus_45.jpg", dummy_img2, "image/jpeg"))
    ]
    data = {"metadata": json.dumps(metadata)}
    
    res = client.post("/api/v1/vision/query", data=data, files=files)
    assert res.status_code == 200
    resp_json = res.json()
    assert resp_json["intent"] == "SELECT_OBJECT"
    assert resp_json["requires_segmentation"] is True
    assert resp_json["target"] is not None
    assert "chair" in resp_json["target"]["label"]
    assert resp_json["segmentation"] is not None
    assert len(resp_json["segmentation"]["views"]) >= 1
    first_view = resp_json["segmentation"]["views"][0]
    assert first_view["mask_png_base64"].startswith("data:image/png;base64,")
    assert first_view["foreground_pixel_count"] > 0
    assert first_view["sam_confidence"] is not None
