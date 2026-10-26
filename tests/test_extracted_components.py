import json
from PIL import Image
from app.services.image_preprocessor import ImagePreprocessor
from app.services.prompt_engine import ScenePromptEngine
from app.services.response_parser import StructuredResponseParser
from app.prompts.scene_analysis import SYSTEM_INSTRUCTION, build_user_prompt
from app.schemas.vision import IntentType, NormalizedBBox

def test_image_preprocessor_resizing():
    preprocessor = ImagePreprocessor(max_dimension=384)
    # Test image larger than 384
    img_large = Image.new("RGB", (768, 512), color="blue")
    processed = preprocessor.preprocess_images({"view1": img_large})
    assert "view1" in processed
    p_img = processed["view1"]
    assert max(p_img.width, p_img.height) == 384
    # Check aspect ratio preservation: 768 / 512 = 1.5; 384 / 256 = 1.5
    assert p_img.width == 384
    assert p_img.height == 256

    # Test image smaller than 384 (must remain unchanged)
    img_small = Image.new("RGB", (200, 150), color="red")
    processed_small = preprocessor.preprocess_images({"view2": img_small})
    assert processed_small["view2"].size == (200, 150)

def test_scene_prompt_engine_exact_string_match():
    engine = ScenePromptEngine()
    assert engine.get_system_instruction() == SYSTEM_INSTRUCTION

    question = "Select the brown armchair."
    scope = "scene"
    views = [{"view_id": "current", "role": "active"}, {"view_id": "yaw_45", "role": "orbit"}]
    roi = {"x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.9}
    history = [{"role": "user", "content": "Hello"}]

    legacy_prompt = build_user_prompt(question, scope, views, roi, history)
    engine_prompt = engine.build_user_prompt(question, scope, views, roi, history)
    assert engine_prompt == legacy_prompt

def test_structured_response_parser_valid_json():
    parser = StructuredResponseParser()
    raw = json.dumps({
        "intent": "SELECT_OBJECT",
        "requires_segmentation": True,
        "target": {"label": "chair", "description": "brown armchair"},
        "grounded_views": [
            {
                "view_id": "current",
                "visible": True,
                "bbox": {"x_min": 0.2, "y_min": 0.3, "x_max": 0.6, "y_max": 0.8},
                "confidence": 0.95
            }
        ],
        "answer": "Found the brown armchair in the current view.",
        "observations": [
            {
                "label": "chair",
                "view_id": "current",
                "bbox": {"x_min": 0.2, "y_min": 0.3, "x_max": 0.6, "y_max": 0.8},
                "confidence": 0.95,
                "description": "Armchair contour"
            }
        ]
    })
    res = parser.parse_response(raw, query_id="test1", question="Select the chair")
    assert res.intent == IntentType.SELECT_OBJECT
    assert res.requires_segmentation is True
    assert res.target.label == "chair"
    assert len(res.observations) == 1
    assert res.observations[0].bbox.x_min == 0.2

def test_structured_response_parser_markdown_fence_and_repair():
    parser = StructuredResponseParser()
    raw = """```json
    {
      "intent": "VISUAL_QUERY",
      "answer": "The room contains a desk and lamp.",
      "observations": [
        {
          "label": "lamp",
          "view_id": "current",
          "box_2d": [100, 200, 400, 600],
          "confidence": 0.9
        },
      ]
    }
    ```"""
    res = parser.parse_response(raw, query_id="test2", question="What is in the room?")
    assert res.intent == IntentType.VISUAL_QUERY
    assert res.requires_segmentation is False
    assert "desk and lamp" in res.answer
    assert len(res.observations) == 1
    # box_2d [ymin=100, xmin=200, ymax=400, xmax=600] scaled 0..1000 -> x_min=0.2, y_min=0.1
    assert abs(res.observations[0].bbox.x_min - 0.2) < 1e-4
    assert abs(res.observations[0].bbox.y_min - 0.1) < 1e-4

def test_structured_response_parser_bounding_box_formats():
    parser = StructuredResponseParser()
    # NormalizedBBox direct
    b0 = NormalizedBBox(x_min=0.1, y_min=0.2, x_max=0.8, y_max=0.9)
    assert parser.extract_bounding_box(b0) == b0

    # Dict x_min/y_min/x_max/y_max
    b1 = parser.extract_bounding_box({"x_min": 0.1, "y_min": 0.2, "x_max": 0.8, "y_max": 0.9})
    assert b1.x_min == 0.1 and b1.y_max == 0.9

    # Dict 0..1000 scale
    b2 = parser.extract_bounding_box({"x_min": 100, "y_min": 200, "x_max": 800, "y_max": 900})
    assert b2.x_min == 0.1 and b2.y_max == 0.9

    # List [ymin, xmin, ymax, xmax] 0..1
    b3 = parser.extract_bounding_box([0.2, 0.1, 0.9, 0.8])
    assert b3.x_min == 0.1 and b3.y_min == 0.2 and b3.x_max == 0.8 and b3.y_max == 0.9
