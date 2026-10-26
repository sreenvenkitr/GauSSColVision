import pytest
from app.services.visual_reasoning import VisualReasoningService
from app.models.gemini import MockVisionModel
from app.schemas.vision import VisionQueryRequest, QueryScope, ViewMetadata
from PIL import Image

def test_robust_json_parsing_clean():
    service = VisualReasoningService(model=MockVisionModel())
    raw = '{"answer": "Clean test answer", "observations": [], "possible_explanations": [], "uncertainty": null, "recommended_next_view": {"required": false}}'
    resp = service._robust_parse_response(raw, "test_1")
    assert resp.answer == "Clean test answer"
    assert len(resp.observations) == 0

def test_robust_json_parsing_fenced():
    service = VisualReasoningService(model=MockVisionModel())
    raw = '''Here is your inspection analysis:
```json
{
  "answer": "Detected fracture near flange.",
  "observations": [
    {
      "label": "fracture",
      "view_id": "current",
      "bbox": {"x_min": 0.3, "y_min": 0.4, "x_max": 0.6, "y_max": 0.7},
      "confidence": 0.85,
      "description": "Surface crack"
    }
  ],
  "possible_explanations": ["impact", "fatigue"],
  "uncertainty": "Lighting casts slight shadow.",
  "recommended_next_view": {
    "required": true,
    "reason": "Closer angle needed",
    "yaw_delta_deg": 20.0,
    "pitch_delta_deg": -10.0,
    "zoom_factor": 1.5
  }
}
```
Hope this helps!'''
    resp = service._robust_parse_response(raw, "test_2")
    assert resp.answer == "Detected fracture near flange."
    assert len(resp.observations) == 1
    assert resp.observations[0].label == "fracture"
    assert resp.observations[0].bbox.x_min == 0.3
    assert resp.recommended_next_view.required is True
    assert resp.recommended_next_view.yaw_delta_deg == 20.0

def test_robust_json_parsing_malformed_fallback():
    service = VisualReasoningService(model=MockVisionModel())
    raw = "The object appears to be a normal room without any visible structural failure."
    resp = service._robust_parse_response(raw, "test_3")
    assert "The object appears to be a normal room" in resp.answer
    assert len(resp.observations) == 0
    assert resp.uncertainty is not None
