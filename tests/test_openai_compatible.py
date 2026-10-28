import pytest
from unittest.mock import patch, MagicMock
from PIL import Image

from app.models.openai_compatible import OpenAICompatibleVLM
from app.schemas.vision import VisionQueryRequest, QueryScope, ViewMetadata


def test_openai_compatible_initialization():
    model = OpenAICompatibleVLM(
        api_key="sk-or-test12345",
        model_name="qwen/qwen-2-vl-72b-instruct:free",
        base_url="https://openrouter.ai/api/v1"
    )
    assert model.ready is True
    info = model.get_info()
    assert "qwen" in info["model"]
    assert info["base_url"] == "https://openrouter.ai/api/v1"


def test_openai_compatible_extract_content():
    model = OpenAICompatibleVLM(api_key="test-key")
    sample_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "<think>Thinking about the armchair...</think>{\"answer\": \"Found the chair\"}"
                }
            }
        ]
    }
    extracted = model._extract_content_from_response(sample_response)
    assert "<think>" not in extracted
    assert '{"answer": "Found the chair"}' in extracted


@patch("requests.post")
def test_openai_compatible_analyze(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '{"intent": "SELECT_OBJECT", "requires_segmentation": true, "target": {"label": "chair"}, "grounded_views": [{"view_id": "current", "visible": true, "bbox": {"x_min": 0.2, "y_min": 0.3, "x_max": 0.7, "y_max": 0.8}, "confidence": 0.9}], "answer": "Chair selected", "observations": []}'
                }
            }
        ]
    }
    mock_post.return_value = mock_resp

    model = OpenAICompatibleVLM(api_key="test-key", model_name="qwen/qwen-2-vl-72b-instruct:free")
    req = VisionQueryRequest(
        question="Select the chair",
        query_scope=QueryScope.CURRENT_VIEW,
        views=[ViewMetadata(view_id="current", role="user_viewport")]
    )
    images = {"current": Image.new("RGB", (64, 64), color="blue")}

    result_text = model.analyze(req, images)
    assert "Chair selected" in result_text
    assert mock_post.called
