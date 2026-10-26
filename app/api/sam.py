import io
import logging
from fastapi import APIRouter, Form, File, UploadFile, HTTPException, Depends
from PIL import Image

from ..schemas.vision import NormalizedBBox, SAMSegmentResponse
from ..services.sam2_service import SAM2Service, get_sam2_service

logger = logging.getLogger("gausscol_vision.api.sam")
router = APIRouter(tags=["SAM Segmentation"])

@router.post("/api/v1/sam/segment", response_model=SAMSegmentResponse)
@router.post("/api/segment", response_model=SAMSegmentResponse)
async def sam_segment(
    image: UploadFile = File(..., description="Image to segment"),
    x_min: float = Form(..., ge=0.0, le=1.0, description="Normalized left [0, 1]"),
    y_min: float = Form(..., ge=0.0, le=1.0, description="Normalized top [0, 1]"),
    x_max: float = Form(..., ge=0.0, le=1.0, description="Normalized right [0, 1]"),
    y_max: float = Form(..., ge=0.0, le=1.0, description="Normalized bottom [0, 1]"),
    service: SAM2Service = Depends(get_sam2_service)
):
    """
    Direct SAM 2.1 debug endpoint.
    Given an image and bounding box, runs SAM 2.1 Hiera Tiny and returns
    the binary segmentation mask and confidence score without involving Gemini.
    """
    try:
        content = await image.read()
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode uploaded image: {e}")

    bbox = NormalizedBBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

    try:
        from starlette.concurrency import run_in_threadpool
        result = await run_in_threadpool(service.segment_image, pil_img, bbox)
        return SAMSegmentResponse(
            status="ok",
            confidence=result["confidence"],
            width=result["width"],
            height=result["height"],
            foreground_pixel_count=result["foreground_pixel_count"],
            mask_png_base64=result["mask_png_base64"],
            timing_ms=result["timing_ms"]
        )
    except Exception as err:
        logger.error(f"SAM segmentation failed: {err}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"SAM segmentation error: {str(err)}")
