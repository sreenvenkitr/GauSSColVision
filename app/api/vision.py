import json
import io
import logging
from typing import List, Optional
from fastapi import APIRouter, Form, File, UploadFile, HTTPException, Depends
from PIL import Image

from ..schemas.vision import VisionQueryRequest, VisionQueryResponse, HealthResponse
from ..services.visual_reasoning import VisualReasoningService
from ..models.gemini import get_vlm_model

logger = logging.getLogger("gausscol_vision.api")
router = APIRouter()

def get_reasoning_service():
    return VisualReasoningService()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    from ..services.sam2_service import get_sam2_service
    model = get_vlm_model()
    info = model.get_info()
    sam_info = get_sam2_service().get_info()
    return HealthResponse(
        status="ok",
        model=info.get("model", "unknown"),
        device=info.get("device", "cpu"),
        cuda_available=info.get("cuda_available", False),
        dtype=info.get("dtype", "float32"),
        ready=info.get("ready", False),
        sam_enabled=sam_info.get("enabled", True),
        sam_model=sam_info.get("model", "sam2.1_hiera_tiny"),
        sam_device=sam_info.get("device", "cpu"),
        sam_ready=sam_info.get("ready", False)
    )

@router.post("/api/v1/vision/query", response_model=VisionQueryResponse)
async def vision_query(
    metadata: str = Form(..., description="JSON-serialized VisionQueryRequest"),
    images: List[UploadFile] = File(default=[], description="Image files for each view"),
    service: VisualReasoningService = Depends(get_reasoning_service)
):
    # 1. Parse JSON metadata
    try:
        req_dict = json.loads(metadata)
        request = VisionQueryRequest.model_validate(req_dict) if hasattr(VisionQueryRequest, "model_validate") else VisionQueryRequest.parse_obj(req_dict)
    except Exception as err:
        logger.error(f"Failed to parse query metadata: {err}")
        raise HTTPException(status_code=422, detail=f"Invalid query metadata: {str(err)}")
        
    # 2. Decode uploaded images
    pil_images = {}
    for idx, uploaded in enumerate(images):
        try:
            content = await uploaded.read()
            img = Image.open(io.BytesIO(content)).convert("RGB")
            
            # Associate view_id: use filename without extension or match to request.views
            view_id = uploaded.filename.rsplit(".", 1)[0] if uploaded.filename else ""
            if not view_id or view_id in pil_images:
                if idx < len(request.views):
                    view_id = request.views[idx].view_id
                else:
                    view_id = f"view_{idx}"
                    
            pil_images[view_id] = img
        except Exception as img_err:
            logger.warning(f"Failed to decode image {uploaded.filename}: {img_err}")
            
    # Ensure request.views accounts for any images received without explicit view metadata
    existing_view_ids = {v.view_id for v in request.views}
    for vid in pil_images.keys():
        if vid not in existing_view_ids:
            from ..schemas.vision import ViewMetadata
            request.views.append(ViewMetadata(view_id=vid))
            
    if not pil_images and request.query_scope != "scene":
        logger.warning("Query received without any valid images.")
        
    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(service.process_query, request, pil_images)
