import pytest
from app.schemas.vision import (
    QueryScope,
    NormalizedBBox,
    VisionObservation,
    RecommendedNextView,
    VisionQueryRequest,
    VisionQueryResponse,
    ViewMetadata
)

def test_normalized_bbox_validation():
    bbox = NormalizedBBox(x_min=0.1, y_min=0.2, x_max=0.8, y_max=0.9)
    assert bbox.x_min == 0.1
    assert bbox.y_max == 0.9

def test_query_scope_enum():
    assert QueryScope.SCENE.value == "scene"
    assert QueryScope.CURRENT_VIEW.value == "current_view"
    assert QueryScope.ROI_2D.value == "roi_2d"
    assert QueryScope.ROI_3D.value == "roi_3d"

def test_vision_query_request_parsing():
    req = VisionQueryRequest(
        question="Where is the chair?",
        query_scope=QueryScope.CURRENT_VIEW,
        views=[ViewMetadata(view_id="current")]
    )
    assert req.question == "Where is the chair?"
    assert len(req.views) == 1
    assert req.views[0].view_id == "current"

def test_vision_query_response_serialization():
    resp = VisionQueryResponse(
        answer="A chair is located on the left.",
        observations=[
            VisionObservation(
                label="chair",
                view_id="current",
                bbox=NormalizedBBox(x_min=0.2, y_min=0.3, x_max=0.5, y_max=0.7),
                confidence=0.92,
                description="Armchair"
            )
        ],
        possible_explanations=[],
        uncertainty=None,
        recommended_next_view=RecommendedNextView(required=False)
    )
    data = resp.model_dump() if hasattr(resp, "model_dump") else resp.dict()
    assert data["answer"] == "A chair is located on the left."
    assert len(data["observations"]) == 1
    assert data["observations"][0]["bbox"]["x_min"] == 0.2
