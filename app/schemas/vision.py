from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class QueryScope(str, Enum):
    SCENE = "scene"
    CURRENT_VIEW = "current_view"
    ROI_2D = "roi_2d"
    ROI_3D = "roi_3d"

class NormalizedBBox(BaseModel):
    x_min: float = Field(..., ge=0.0, le=1.0, description="Normalized left coordinate [0, 1]")
    y_min: float = Field(..., ge=0.0, le=1.0, description="Normalized top coordinate [0, 1]")
    x_max: float = Field(..., ge=0.0, le=1.0, description="Normalized right coordinate [0, 1]")
    y_max: float = Field(..., ge=0.0, le=1.0, description="Normalized bottom coordinate [0, 1]")

class VisionObservation(BaseModel):
    label: str = Field(..., description="Short descriptive label of the detected region/object/defect")
    view_id: str = Field(..., description="Identifier of the view where this observation is best visible")
    bbox: Optional[NormalizedBBox] = Field(None, description="Normalized bounding box if localizable")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Model confidence score [0, 1]")
    description: Optional[str] = Field(None, description="Detailed visual description supporting the observation")

class RecommendedNextView(BaseModel):
    required: bool = Field(False, description="Whether a better camera viewpoint is recommended")
    reason: Optional[str] = Field(None, description="Why the recommended view is needed")
    yaw_delta_deg: Optional[float] = Field(None, description="Suggested relative yaw adjustment in degrees")
    pitch_delta_deg: Optional[float] = Field(None, description="Suggested relative pitch adjustment in degrees")
    zoom_factor: Optional[float] = Field(None, description="Suggested relative zoom magnification (>1 zoom in, <1 zoom out)")

class CameraMetadata(BaseModel):
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    distance: Optional[float] = None
    target: Optional[List[float]] = None
    fov_y: Optional[float] = None

class ViewMetadata(BaseModel):
    view_id: str = Field(..., description="Unique ID of the view, e.g. current, left_60, right_60, roi_detail")
    role: Optional[str] = Field("standard", description="Role of view: context, roi_detail, scene_view, recommended")
    camera_metadata: Optional[CameraMetadata] = None

class ChatTurn(BaseModel):
    role: str = Field(..., description="user or assistant")
    content: str = Field(...)

class VisionQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question about the scene/view/ROI")
    query_scope: QueryScope = Field(QueryScope.CURRENT_VIEW, description="Scope of the query")
    views: List[ViewMetadata] = Field(default_factory=list, description="Metadata for each rendered image view")
    roi: Optional[NormalizedBBox] = Field(None, description="User-selected 2D ROI if query_scope is roi_2d")
    conversation_history: List[ChatTurn] = Field(default_factory=list, description="Recent conversation turns (bounded)")

class IntentType(str, Enum):
    VISUAL_QUERY = "VISUAL_QUERY"
    SELECT_OBJECT = "SELECT_OBJECT"
    CREATE_SCENE_NODE = "CREATE_SCENE_NODE"
    HIDE_OBJECT = "HIDE_OBJECT"
    SHOW_OBJECT = "SHOW_OBJECT"
    DELETE_OBJECT = "DELETE_OBJECT"
    EDIT_OBJECT = "EDIT_OBJECT"

class TargetObject(BaseModel):
    label: str = Field(..., description="Target object class or label, e.g. chair, table, bucket")
    description: Optional[str] = Field(None, description="Detailed attributes, e.g. brown chair, red bucket")

class GroundedView(BaseModel):
    view_id: str = Field(..., description="View ID where object is evaluated")
    visible: bool = Field(True, description="Whether target is visible in this view")
    bbox: Optional[NormalizedBBox] = Field(None, description="Bounding box if visible")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")

class SAMMaskResult(BaseModel):
    view_id: str = Field(..., description="View identifier matching the camera frame")
    gemini_confidence: Optional[float] = Field(None, description="Gemini grounding confidence")
    sam_confidence: Optional[float] = Field(None, description="SAM 2.1 mask predicted IoU score")
    bbox: Optional[NormalizedBBox] = Field(None, description="Bounding box used for prompting SAM")
    mask_png_base64: str = Field(..., description="Base64 encoded binary PNG mask image")
    width: int = Field(..., description="Mask width in pixels")
    height: int = Field(..., description="Mask height in pixels")
    foreground_pixel_count: int = Field(0, description="Total foreground mask pixels")

class SegmentationResult(BaseModel):
    views: List[SAMMaskResult] = Field(default_factory=list, description="Per-view segmentation results")

class PerformanceMetrics(BaseModel):
    image_prep_ms: float = 0.0
    prompt_prep_ms: float = 0.0
    inference_ms: float = 0.0
    parse_ms: float = 0.0
    gemini_ms: float = 0.0
    sam_total_ms: float = 0.0
    total_ms: float = 0.0
    device: str = "cpu"
    gpu_memory_allocated_mb: Optional[float] = None

class VisionQueryResponse(BaseModel):
    answer: str = Field(..., description="Primary textual answer to user query")
    intent: IntentType = Field(IntentType.VISUAL_QUERY, description="Classified intent of the user request")
    requires_segmentation: bool = Field(False, description="Whether this request required object segmentation")
    target: Optional[TargetObject] = Field(None, description="Target object metadata if intent is SELECT_OBJECT")
    observations: List[VisionObservation] = Field(default_factory=list, description="Zero or more localized observations")
    possible_explanations: List[str] = Field(default_factory=list, description="Plausible causes or explanations for causal queries")
    uncertainty: Optional[str] = Field(None, description="Explicit statement of visual uncertainty or limitations")
    recommended_next_view: RecommendedNextView = Field(default_factory=RecommendedNextView, description="Active view recommendation")
    segmentation: Optional[SegmentationResult] = Field(None, description="Multi-view segmentation masks if intent is SELECT_OBJECT")
    metrics: Optional[PerformanceMetrics] = None
    debug_info: Optional[Dict[str, Any]] = Field(None, description="Diagnostic telemetry and raw grounding data for debugging")

class SAMSegmentResponse(BaseModel):
    status: str = "ok"
    confidence: float
    width: int
    height: int
    foreground_pixel_count: int
    mask_png_base64: str
    timing_ms: float

class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    device: str
    cuda_available: bool
    dtype: str
    ready: bool
    sam_enabled: bool = True
    sam_model: str = "sam2.1_hiera_tiny"
    sam_device: str = "cpu"
    sam_ready: bool = False
