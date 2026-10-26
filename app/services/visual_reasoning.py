import time
import uuid
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
import torch
from PIL import Image

from ..schemas.vision import (
    VisionQueryRequest,
    VisionQueryResponse,
    VisionObservation,
    RecommendedNextView,
    PerformanceMetrics,
    NormalizedBBox,
    IntentType,
    TargetObject,
    SAMMaskResult,
    SegmentationResult
)
from ..models.base import VisionLanguageModel
from ..models.gemini import get_vlm_model
from ..config import settings
from .image_preprocessor import ImagePreprocessor
from .prompt_engine import ScenePromptEngine
from .response_parser import StructuredResponseParser

logger = logging.getLogger("gausscol_vision.reasoning")

def _extract_bounding_box(bbox_data: Any) -> Optional[NormalizedBBox]:
    """
    Backward-compatibility function delegating to StructuredResponseParser.
    Preserves module-level import contract for tests and external scripts.
    """
    return StructuredResponseParser().extract_bounding_box(bbox_data)

class VisualReasoningService:
    """
    Main orchestrator for multimodal visual queries and 2D-to-3D object selection.

    Why this class exists:
    Coordinates the pipeline connecting WebGPU multi-view camera renders to Gemini VLM reasoning
    and SAM 2.1 neural segmentation. It preserves the exact execution sequence:
        Multi-View Renders -> Preprocessing -> VLM Query -> Parsing -> Multi-View SAM -> Response
    """

    def __init__(
        self,
        model: Optional[VisionLanguageModel] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
        prompt_engine: Optional[ScenePromptEngine] = None,
        response_parser: Optional[StructuredResponseParser] = None
    ):
        self.model = model or get_vlm_model()
        self.preprocessor = preprocessor or ImagePreprocessor(max_dimension=384)
        self.prompt_engine = prompt_engine or ScenePromptEngine()
        self.response_parser = response_parser or StructuredResponseParser()

    def process_query(self, request: VisionQueryRequest, images: Dict[str, Image.Image]) -> VisionQueryResponse:
        """
        Executes the end-to-end multimodal reasoning pipeline for the incoming request and images.
        """
        query_id = str(uuid.uuid4())[:8]
        start_total = time.time()

        # Log request metadata without serializing image bytes
        image_dims = {k: f"{v.width}x{v.height}" for k, v in images.items()}
        logger.info(
            f"[QUERY:{query_id}] Received query - Scope: {request.query_scope.value}, "
            f"Views: {list(images.keys())}, Dimensions: {image_dims}, "
            f"Question: '{request.question}'"
        )

        # 1. Image preprocessing: Downscale images for VLM throughput
        processed_images, image_prep_ms = self._prepare_images(images)

        # 2. Prompt preparation (legacy timer metric retained for telemetry compatibility)
        prompt_prep_ms = 0.0

        # 3. Model inference: Call Gemini / VLM
        raw_response, gemini_ms, error_response = self._call_vlm(request, processed_images, query_id, start_total, image_prep_ms, prompt_prep_ms)
        if error_response:
            return error_response

        # 4. Response parsing: Extract structured JSON and intent
        parsed_response, parse_ms = self._parse_vlm_response(raw_response, query_id, request.question)
        raw_data = getattr(parsed_response, "_raw_data", {})

        # 5. Multi-View SAM 2.1 Segmentation (if intent is SELECT_OBJECT)
        sam_total_ms, valid_views = self._coordinate_segmentation(
            request=request,
            parsed_response=parsed_response,
            raw_data=raw_data,
            images=images,
            query_id=query_id
        )

        # 6. Assemble telemetry and debug metrics
        total_ms = (time.time() - start_total) * 1000.0
        self._attach_telemetry(
            parsed_response=parsed_response,
            query_id=query_id,
            images=images,
            valid_views=valid_views,
            image_prep_ms=image_prep_ms,
            prompt_prep_ms=prompt_prep_ms,
            gemini_ms=gemini_ms,
            parse_ms=parse_ms,
            sam_total_ms=sam_total_ms,
            total_ms=total_ms
        )

        return parsed_response

    def _prepare_images(self, images: Dict[str, Image.Image]) -> Tuple[Dict[str, Image.Image], float]:
        """
        Downscales image views to max 384px while preserving aspect ratio.
        """
        t0 = time.time()
        processed = self.preprocessor.preprocess_images(images)
        elapsed_ms = (time.time() - t0) * 1000.0
        return processed, elapsed_ms

    def _call_vlm(
        self,
        request: VisionQueryRequest,
        processed_images: Dict[str, Image.Image],
        query_id: str,
        start_total: float,
        image_prep_ms: float,
        prompt_prep_ms: float
    ) -> Tuple[str, float, Optional[VisionQueryResponse]]:
        """
        Dispatches multimodal analysis to the VLM provider and handles transport errors.
        """
        t0 = time.time()
        try:
            raw_response = self.model.analyze(request, processed_images)
            gemini_ms = (time.time() - t0) * 1000.0
            logger.info(f"[QUERY:{query_id}] Model inference completed in {gemini_ms:.1f}ms")
            return raw_response, gemini_ms, None
        except Exception as e:
            gemini_ms = (time.time() - t0) * 1000.0
            logger.error(f"[QUERY:{query_id}] Model inference error: {e}", exc_info=True)
            error_resp = VisionQueryResponse(
                answer=f"Error executing vision reasoning: {str(e)}",
                intent=IntentType.VISUAL_QUERY,
                requires_segmentation=False,
                observations=[],
                possible_explanations=[],
                uncertainty="Inference failure",
                recommended_next_view=RecommendedNextView(required=False),
                metrics=PerformanceMetrics(
                    image_prep_ms=image_prep_ms,
                    prompt_prep_ms=prompt_prep_ms,
                    inference_ms=gemini_ms,
                    gemini_ms=gemini_ms,
                    total_ms=(time.time() - start_total) * 1000.0,
                    device=self.model.get_info().get("device", "unknown")
                )
            )
            return "", gemini_ms, error_resp

    def _parse_vlm_response(self, raw_response: str, query_id: str, question: str) -> Tuple[VisionQueryResponse, float]:
        """
        Delegates text cleanup and JSON parsing to StructuredResponseParser.
        """
        t0 = time.time()
        logger.info(f"[QUERY:{query_id}] Model raw response:\n{raw_response}")
        parsed = self.response_parser.parse_response(raw_response, query_id, question)
        parse_ms = (time.time() - t0) * 1000.0
        return parsed, parse_ms

    def _coordinate_segmentation(
        self,
        request: VisionQueryRequest,
        parsed_response: VisionQueryResponse,
        raw_data: dict,
        images: Dict[str, Image.Image],
        query_id: str
    ) -> Tuple[float, List[Tuple[str, float, NormalizedBBox]]]:
        """
        Orchestrates multi-view SAM 2.1 segmentation when the intent is SELECT_OBJECT.
        """
        valid_views: List[Tuple[str, float, NormalizedBBox]] = []
        sam_total_ms = 0.0

        if not (parsed_response.intent == IntentType.SELECT_OBJECT and parsed_response.requires_segmentation):
            return sam_total_ms, valid_views

        from .sam2_service import get_sam2_service

        if not settings.sam_enabled:
            logger.info(f"[QUERY:{query_id}] SAM is disabled by configuration.")
            parsed_response.answer = "Object identified, but segmentation service is unavailable (SAM_ENABLED=false)."
            parsed_response.requires_segmentation = False
            return sam_total_ms, valid_views

        sam_service = get_sam2_service()

        # Extract target label
        target_label = self._extract_target_label(request.question, parsed_response)

        # Collect grounded view candidates meeting confidence thresholds
        valid_views = self._filter_segmentation_views(raw_data, parsed_response, images, query_id)

        if not valid_views:
            logger.warning(f"[QUERY:{query_id}] No views satisfied grounding threshold for '{target_label}'.")
            parsed_response.answer = f"I could not confidently locate '{target_label}' in the available views."
            parsed_response.requires_segmentation = False
            return sam_total_ms, valid_views

        # Sort viewpoints so the user's active viewport ('current') is always processed first (index 0)
        valid_views.sort(key=lambda x: 0 if x[0] == "current" else 1)
        logger.info(
            f"[QUERY:{query_id}] Running SAM 2.1 across {len(valid_views)} grounded view(s) for '{target_label}'..."
        )

        sam_start = time.time()
        sam_mask_results: List[SAMMaskResult] = []

        for vid, gem_conf, bbox_obj in valid_views:
            orig_img = images[vid]
            try:
                sam_res = sam_service.segment_image(orig_img, bbox_obj)
                if sam_res["foreground_pixel_count"] == 0:
                    logger.warning(f"[QUERY:{query_id}] SAM returned empty mask for view '{vid}'.")
                    continue

                sam_mask_results.append(SAMMaskResult(
                    view_id=vid,
                    gemini_confidence=gem_conf,
                    sam_confidence=sam_res["confidence"],
                    bbox=bbox_obj,
                    mask_png_base64=sam_res["mask_png_base64"],
                    width=sam_res["width"],
                    height=sam_res["height"],
                    foreground_pixel_count=sam_res["foreground_pixel_count"]
                ))
            except Exception as sam_err:
                logger.error(f"[QUERY:{query_id}] SAM error on view '{vid}': {sam_err}")

        sam_total_ms = (time.time() - sam_start) * 1000.0

        if sam_mask_results:
            parsed_response.segmentation = SegmentationResult(views=sam_mask_results)
            parsed_response.answer = f"Selected '{target_label}' across {len(sam_mask_results)} supporting viewpoints."
        else:
            parsed_response.answer = f"Object '{target_label}' identified, but SAM mask generation produced zero foreground pixels."
            parsed_response.requires_segmentation = False

        return sam_total_ms, valid_views

    def _extract_target_label(self, question: str, parsed_response: VisionQueryResponse) -> str:
        """
        Extracts a clean, human-readable label for the target object from model output or the user question.
        """
        if parsed_response.target and parsed_response.target.label and parsed_response.target.label.lower() not in ("object", "target object", "target"):
            return parsed_response.target.label

        cleaned_q = re.sub(r"^(select|highlight|segment|isolate|pick|find)\s+(the\s+)?", "", question, flags=re.I).strip()
        cleaned_q = re.sub(r"\s+(in\s+the\s+(scene|room|view)|from\s+the\s+scene).*$", "", cleaned_q, flags=re.I).strip()
        if cleaned_q:
            if parsed_response.target:
                parsed_response.target.label = cleaned_q
                parsed_response.target.description = cleaned_q
            else:
                parsed_response.target = TargetObject(label=cleaned_q, description=cleaned_q)
            return cleaned_q

        return "object"

    def _filter_segmentation_views(
        self,
        raw_data: dict,
        parsed_response: VisionQueryResponse,
        images: Dict[str, Image.Image],
        query_id: str
    ) -> List[Tuple[str, float, NormalizedBBox]]:
        """
        Identifies and validates candidate viewpoints containing bounding boxes for SAM segmentation.
        """
        # Gather candidate entries from grounded_views, views, observations, or parsed observations
        grounded_entries = raw_data.get("grounded_views", [])
        if not grounded_entries and "views" in raw_data:
            grounded_entries = raw_data["views"]
        if not grounded_entries and "observations" in raw_data:
            grounded_entries = raw_data["observations"]
        if not grounded_entries and parsed_response.observations:
            grounded_entries = [
                {
                    "view_id": obs.view_id,
                    "visible": True,
                    "confidence": obs.confidence if obs.confidence is not None else 0.85,
                    "bbox": obs.bbox
                }
                for obs in parsed_response.observations
                if obs.bbox is not None
            ]

        valid_views: List[Tuple[str, float, NormalizedBBox]] = []
        for entry in grounded_entries:
            vid = str(entry.get("view_id", ""))
            vis = bool(entry.get("visible", True))
            conf = float(entry.get("confidence", 0.0)) if entry.get("confidence") is not None else 0.85
            raw_bbox = entry.get("bbox") or entry.get("box_2d")
            bbox_obj = self.response_parser.extract_bounding_box(raw_bbox)

            if vid in images and vis and conf >= settings.gemini_grounding_min_confidence and bbox_obj:
                valid_views.append((vid, conf, bbox_obj))

        # Fallback: If strict confidence threshold filtered all views, accept any viewpoint with a valid bbox
        if not valid_views and grounded_entries:
            logger.info(f"[QUERY:{query_id}] Threshold {settings.gemini_grounding_min_confidence} filtered all views; trying fallback...")
            for entry in grounded_entries:
                vid = str(entry.get("view_id", ""))
                vis = bool(entry.get("visible", True))
                conf = float(entry.get("confidence", 0.0)) if entry.get("confidence") is not None else 0.85
                raw_bbox = entry.get("bbox") or entry.get("box_2d")
                bbox_obj = self.response_parser.extract_bounding_box(raw_bbox)
                if vid in images and vis and bbox_obj:
                    valid_views.append((vid, conf, bbox_obj))

        return valid_views

    def _attach_telemetry(
        self,
        parsed_response: VisionQueryResponse,
        query_id: str,
        images: Dict[str, Image.Image],
        valid_views: List[Tuple[str, float, NormalizedBBox]],
        image_prep_ms: float,
        prompt_prep_ms: float,
        gemini_ms: float,
        parse_ms: float,
        sam_total_ms: float,
        total_ms: float
    ) -> None:
        """
        Populates PerformanceMetrics and debug telemetry matching existing API contracts.
        """
        gpu_mem = None
        if torch.cuda.is_available():
            try:
                gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)
            except Exception:
                gpu_mem = None

        parsed_response.metrics = PerformanceMetrics(
            image_prep_ms=round(image_prep_ms, 2),
            prompt_prep_ms=round(prompt_prep_ms, 2),
            inference_ms=round(gemini_ms, 2),
            parse_ms=round(parse_ms, 2),
            gemini_ms=round(gemini_ms, 2),
            sam_total_ms=round(sam_total_ms, 2),
            total_ms=round(total_ms, 2),
            device=self.model.get_info().get("device", "cpu"),
            gpu_memory_allocated_mb=round(gpu_mem, 1) if gpu_mem else None
        )

        target_dict = None
        if parsed_response.target:
            target_dict = parsed_response.target.model_dump() if hasattr(parsed_response.target, 'model_dump') else parsed_response.target.dict()

        parsed_response.debug_info = {
            "model": self.model.get_info().get("name", "Gemini"),
            "intent": parsed_response.intent.value,
            "target": target_dict,
            "views_sent": list(images.keys()),
            "views_grounded_count": len(valid_views),
            "grounded_views": [v[0] for v in valid_views],
            "gemini_ms": round(gemini_ms, 1),
            "sam_ms": round(sam_total_ms, 1),
            "total_ms": round(total_ms, 1),
            "sam_mask_count": len(parsed_response.segmentation.views) if parsed_response.segmentation else 0
        }

        logger.info(
            f"[QUERY:{query_id}] Finished in {total_ms:.1f}ms. "
            f"Intent: {parsed_response.intent.value}, "
            f"Segmentation: {bool(parsed_response.segmentation and parsed_response.segmentation.views)}"
        )
        logger.info(f"[QUERY:{query_id}] === ANSWER ===")
        logger.info(f"[QUERY:{query_id}] {parsed_response.answer}")
        if parsed_response.segmentation and parsed_response.segmentation.views:
            logger.info(f"[QUERY:{query_id}] === SAM SEGMENTATION ({len(parsed_response.segmentation.views)} views) ===")
            for sm in parsed_response.segmentation.views:
                logger.info(f"[QUERY:{query_id}] - [{sm.view_id}] sam_conf={sm.sam_confidence}, fg_pixels={sm.foreground_pixel_count}")
        logger.info(f"[QUERY:{query_id}] ====================")

    def _robust_parse_response(self, raw_text: str, query_id: str, question: str = "") -> VisionQueryResponse:
        """
        Backward-compatibility method delegating to StructuredResponseParser.
        """
        return self.response_parser.parse_response(raw_text, query_id, question)
