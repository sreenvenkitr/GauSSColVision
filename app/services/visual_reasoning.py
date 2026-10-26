import time
import json
import re
import uuid
import logging
from typing import Dict, Any, List, Optional
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

logger = logging.getLogger("gausscol_vision.reasoning")

def _extract_bounding_box(bbox_data: Any) -> Optional[NormalizedBBox]:
    """
    Robustly extracts a NormalizedBBox from multiple formats:
    - NormalizedBBox instance
    - Dict with x_min/y_min/x_max/y_max or xmin/ymin/xmax/ymax
    - Dict with box_2d: [ymin, xmin, ymax, xmax]
    - List/tuple [ymin, xmin, ymax, xmax] (Gemini standard)
    - Auto-scales if integer coordinates 0..1000 are used
    """
    if not bbox_data:
        return None

    if isinstance(bbox_data, NormalizedBBox):
        return bbox_data

    # Format 1: dict
    if isinstance(bbox_data, dict):
        if "box_2d" in bbox_data:
            return _extract_bounding_box(bbox_data["box_2d"])

        x_min = bbox_data.get("x_min", bbox_data.get("xmin"))
        y_min = bbox_data.get("y_min", bbox_data.get("ymin"))
        x_max = bbox_data.get("x_max", bbox_data.get("xmax"))
        y_max = bbox_data.get("y_max", bbox_data.get("ymax"))

        if None not in (x_min, y_min, x_max, y_max):
            try:
                fx_min, fy_min, fx_max, fy_max = float(x_min), float(y_min), float(x_max), float(y_max)
                if max(fx_min, fy_min, fx_max, fy_max) > 1.0:
                    fx_min /= 1000.0
                    fy_min /= 1000.0
                    fx_max /= 1000.0
                    fy_max /= 1000.0
                return NormalizedBBox(
                    x_min=max(0.0, min(1.0, min(fx_min, fx_max))),
                    y_min=max(0.0, min(1.0, min(fy_min, fy_max))),
                    x_max=max(0.0, min(1.0, max(fx_min, fx_max))),
                    y_max=max(0.0, min(1.0, max(fy_min, fy_max)))
                )
            except (ValueError, TypeError):
                pass

    # Format 2: list/tuple
    if isinstance(bbox_data, (list, tuple)) and len(bbox_data) >= 4:
        try:
            c1, c2, c3, c4 = [float(v) for v in bbox_data[:4]]
            if max(c1, c2, c3, c4) > 1.0:
                c1 /= 1000.0
                c2 /= 1000.0
                c3 /= 1000.0
                c4 /= 1000.0
            # Gemini standard order is [ymin, xmin, ymax, xmax]
            y_min, x_min, y_max, x_max = c1, c2, c3, c4
            return NormalizedBBox(
                x_min=max(0.0, min(1.0, min(x_min, x_max))),
                y_min=max(0.0, min(1.0, min(y_min, y_max))),
                x_max=max(0.0, min(1.0, max(x_min, x_max))),
                y_max=max(0.0, min(1.0, max(y_min, y_max)))
            )
        except (ValueError, TypeError):
            pass

    return None

class VisualReasoningService:
    def __init__(self, model: Optional[VisionLanguageModel] = None):
        self.model = model or get_vlm_model()

    def process_query(self, request: VisionQueryRequest, images: Dict[str, Image.Image]) -> VisionQueryResponse:
        query_id = str(uuid.uuid4())[:8]
        start_total = time.time()
        
        # Log incoming request metadata (no image bytes)
        image_dims = {k: f"{v.width}x{v.height}" for k, v in images.items()}
        logger.info(
            f"[QUERY:{query_id}] Received query - Scope: {request.query_scope.value}, "
            f"Views: {list(images.keys())}, Dimensions: {image_dims}, "
            f"Question: '{request.question}'"
        )
        
        # 1. Image preprocessing: Downscale images for VLM throughput if needed
        t0 = time.time()
        processed_images = {}
        max_dim = 384
        for k, v in images.items():
            if max(v.width, v.height) > max_dim:
                scale = max_dim / float(max(v.width, v.height))
                new_w = max(28, int(v.width * scale))
                new_h = max(28, int(v.height * scale))
                processed_images[k] = v.resize((new_w, new_h), Image.Resampling.BILINEAR)
            else:
                processed_images[k] = v
        image_prep_ms = (time.time() - t0) * 1000.0
        
        # 2. Prompt preparation
        t1 = time.time()
        prompt_prep_ms = (time.time() - t1) * 1000.0
        
        # 3. Model inference (Gemini / VLM)
        t2 = time.time()
        try:
            raw_response = self.model.analyze(request, processed_images)
            gemini_ms = (time.time() - t2) * 1000.0
            logger.info(f"[QUERY:{query_id}] Model inference completed in {gemini_ms:.1f}ms")
        except Exception as e:
            gemini_ms = (time.time() - t2) * 1000.0
            logger.error(f"[QUERY:{query_id}] Model inference error: {e}", exc_info=True)
            return VisionQueryResponse(
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
            
        # 4. Response parsing & robust extraction
        t3 = time.time()
        logger.info(f"[QUERY:{query_id}] Model raw response:\n{raw_response}")
        parsed_response = self._robust_parse_response(raw_response, query_id, request.question)
        raw_data = getattr(parsed_response, "_raw_data", {})
        parse_ms = (time.time() - t3) * 1000.0
        
        # 5. Intent Routing & Multi-View SAM 2.1 Segmentation
        sam_total_ms = 0.0
        if parsed_response.intent == IntentType.SELECT_OBJECT and parsed_response.requires_segmentation:
            from .sam2_service import get_sam2_service
            from ..config import settings
            from ..schemas.vision import SAMMaskResult, SegmentationResult

            if not settings.sam_enabled:
                logger.info(f"[QUERY:{query_id}] SAM is disabled by configuration.")
                parsed_response.answer = "Object identified, but segmentation service is unavailable (SAM_ENABLED=false)."
                parsed_response.requires_segmentation = False
            else:
                sam_service = get_sam2_service()
                
                # Gather candidate views from grounded_views, views, observations, or parsed_response.observations
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

                # Extract target label
                target_label = "object"
                if parsed_response.target and parsed_response.target.label and parsed_response.target.label.lower() not in ("object", "target object", "target"):
                    target_label = parsed_response.target.label
                else:
                    cleaned_q = re.sub(r"^(select|highlight|segment|isolate|pick|find)\s+(the\s+)?", "", request.question, flags=re.I).strip()
                    cleaned_q = re.sub(r"\s+(in\s+the\s+(scene|room|view)|from\s+the\s+scene).*$", "", cleaned_q, flags=re.I).strip()
                    if cleaned_q:
                        target_label = cleaned_q
                        if parsed_response.target:
                            parsed_response.target.label = cleaned_q
                            parsed_response.target.description = cleaned_q
                        else:
                            parsed_response.target = TargetObject(label=cleaned_q, description=cleaned_q)

                # Filter valid candidate views meeting confidence threshold
                valid_views = []
                for entry in grounded_entries:
                    vid = str(entry.get("view_id", ""))
                    vis = bool(entry.get("visible", True))
                    conf = float(entry.get("confidence", 0.0)) if entry.get("confidence") is not None else 0.85
                    raw_bbox = entry.get("bbox") or entry.get("box_2d")
                    bbox_obj = _extract_bounding_box(raw_bbox)

                    if vid in images and vis and conf >= settings.gemini_grounding_min_confidence and bbox_obj:
                        valid_views.append((vid, conf, bbox_obj))

                # Fallback: If strict threshold filtered everything, accept any views that have valid bboxes
                if not valid_views and grounded_entries:
                    logger.info(f"[QUERY:{query_id}] Threshold {settings.gemini_grounding_min_confidence} filtered all views; trying fallback...")
                    for entry in grounded_entries:
                        vid = str(entry.get("view_id", ""))
                        vis = bool(entry.get("visible", True))
                        conf = float(entry.get("confidence", 0.0)) if entry.get("confidence") is not None else 0.85
                        raw_bbox = entry.get("bbox") or entry.get("box_2d")
                        bbox_obj = _extract_bounding_box(raw_bbox)
                        if vid in images and vis and bbox_obj:
                            valid_views.append((vid, conf, bbox_obj))

                if not valid_views:
                    logger.warning(f"[QUERY:{query_id}] No views satisfied grounding threshold for '{target_label}'. Total entries checked: {len(grounded_entries)}")
                    parsed_response.answer = f"I could not confidently locate '{target_label}' in the available views."
                    parsed_response.requires_segmentation = False
                else:
                    # Ensure 'current' view is processed first so primary active view is index 0
                    valid_views.sort(key=lambda x: 0 if x[0] == "current" else 1)
                    logger.info(
                        f"[QUERY:{query_id}] Running SAM 2.1 across {len(valid_views)} grounded view(s) for '{target_label}'..."
                    )
                    sam_start = time.time()
                    sam_mask_results = []

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

        total_ms = (time.time() - start_total) * 1000.0
        
        # GPU memory telemetry if CUDA
        gpu_mem = None
        if torch.cuda.is_available():
            try:
                gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)
            except Exception:
                gpu_mem = None
                
        metrics = PerformanceMetrics(
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
        parsed_response.metrics = metrics
        
        parsed_response.debug_info = {
            "model": self.model.get_info().get("name", "Gemini"),
            "intent": parsed_response.intent.value,
            "target": parsed_response.target.dict() if parsed_response.target else None,
            "views_sent": list(images.keys()),
            "views_grounded_count": len(valid_views) if 'valid_views' in locals() else 0,
            "grounded_views": [v[0] for v in valid_views] if 'valid_views' in locals() else [],
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
        return parsed_response

    def _robust_parse_response(self, raw_text: str, query_id: str, question: str = "") -> VisionQueryResponse:
        """
        Robustly extracts structured JSON from model text.
        Extracts intent, target, observations, and grounded views.
        """
        if not raw_text or not raw_text.strip():
            logger.warning(f"[QUERY:{query_id}] Model returned empty string.")
            resp = VisionQueryResponse(
                answer="No response was generated by the model.",
                intent=IntentType.VISUAL_QUERY,
                requires_segmentation=False,
                observations=[],
                possible_explanations=[],
                uncertainty="Empty model output",
                recommended_next_view=RecommendedNextView(required=False)
            )
            resp._raw_data = {}
            return resp

        text = raw_text.strip()
        
        # 1. Strip markdown code fences if present
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
            else:
                cleaned = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "").strip()
                text = cleaned
                
        # 2. Locate outermost braces
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_candidate = text[start_idx : end_idx + 1]
            try:
                data = json.loads(json_candidate)
                
                # Extract intent
                raw_intent = str(data.get("intent", "")).upper()
                is_select_query = any(w in question.lower() for w in ["select", "highlight", "segment", "isolate", "pick"])
                if raw_intent == "SELECT_OBJECT" or is_select_query:
                    intent = IntentType.SELECT_OBJECT
                    requires_segmentation = True
                else:
                    intent = IntentType.VISUAL_QUERY
                    requires_segmentation = False

                # Extract target
                target_obj = None
                target_data = data.get("target")
                if target_data and isinstance(target_data, dict):
                    t_lbl = str(target_data.get("label", "object")).strip()
                    t_desc = str(target_data.get("description", t_lbl)).strip()
                    if t_lbl.lower() in ("object", "target", "target object", "") and intent == IntentType.SELECT_OBJECT:
                        cleaned_q = re.sub(r"^(select|highlight|segment|isolate|pick|find)\s+(the\s+)?", "", question, flags=re.I).strip()
                        cleaned_q = re.sub(r"\s+(in\s+the\s+(scene|room|view)|from\s+the\s+scene).*$", "", cleaned_q, flags=re.I).strip()
                        if cleaned_q:
                            t_lbl = cleaned_q
                            t_desc = cleaned_q
                    target_obj = TargetObject(label=t_lbl, description=t_desc)
                elif intent == IntentType.SELECT_OBJECT:
                    cleaned_q = re.sub(r"^(select|highlight|segment|isolate|pick|find)\s+(the\s+)?", "", question, flags=re.I).strip()
                    cleaned_q = re.sub(r"\s+(in\s+the\s+(scene|room|view)|from\s+the\s+scene).*$", "", cleaned_q, flags=re.I).strip()
                    target_obj = TargetObject(label=cleaned_q or "object", description=question)

                # Sanitize observations
                obs_list = []
                for item in data.get("observations", []):
                    try:
                        raw_bbox = item.get("bbox") or item.get("box_2d")
                        bbox_obj = _extract_bounding_box(raw_bbox)
                        obs = VisionObservation(
                            label=str(item.get("label", "detected feature")),
                            view_id=str(item.get("view_id", "current")),
                            bbox=bbox_obj,
                            confidence=float(item["confidence"]) if item.get("confidence") is not None else None,
                            description=item.get("description")
                        )
                        obs_list.append(obs)
                    except Exception as parse_item_err:
                        logger.warning(f"[QUERY:{query_id}] Error parsing observation item: {parse_item_err}")
                        
                # Next view
                rec_view_data = data.get("recommended_next_view", {})
                rec_view = RecommendedNextView(
                    required=bool(rec_view_data.get("required", False)),
                    reason=rec_view_data.get("reason"),
                    yaw_delta_deg=rec_view_data.get("yaw_delta_deg"),
                    pitch_delta_deg=rec_view_data.get("pitch_delta_deg"),
                    zoom_factor=rec_view_data.get("zoom_factor")
                )
                
                resp = VisionQueryResponse(
                    answer=str(data.get("answer", text)),
                    intent=intent,
                    requires_segmentation=requires_segmentation,
                    target=target_obj,
                    observations=obs_list,
                    possible_explanations=[str(e) for e in data.get("possible_explanations", [])],
                    uncertainty=data.get("uncertainty"),
                    recommended_next_view=rec_view
                )
                resp._raw_data = data
                return resp
            except Exception as json_err:
                logger.warning(f"[QUERY:{query_id}] JSON decode error on candidate: {json_err}. Attempting repair...")
                repaired = json_candidate
                repaired = re.sub(r",\s*([\]}])", r"\1", repaired)
                if repaired.count('"') % 2 != 0:
                    repaired += '"'
                open_braces = repaired.count("{") - repaired.count("}")
                if open_braces > 0:
                    repaired += "}" * open_braces
                open_brackets = repaired.count("[") - repaired.count("]")
                if open_brackets > 0:
                    repaired += "]" * open_brackets
                try:
                    data = json.loads(repaired)
                    logger.info(f"[QUERY:{query_id}] Successfully repaired JSON candidate.")
                    ans = str(data.get("answer", ""))
                    if ans:
                        is_sel = any(w in question.lower() for w in ["select", "highlight", "segment", "isolate", "pick"])
                        resp = VisionQueryResponse(
                            answer=ans,
                            intent=IntentType.SELECT_OBJECT if is_sel else IntentType.VISUAL_QUERY,
                            requires_segmentation=is_sel,
                            target=TargetObject(label=str(data.get("target", {}).get("label", "object")), description=str(data.get("target", {}).get("description", "object"))) if data.get("target") else None,
                            observations=[],
                            possible_explanations=[str(e) for e in data.get("possible_explanations", [])],
                            uncertainty=data.get("uncertainty"),
                            recommended_next_view=RecommendedNextView(required=False)
                        )
                        resp._raw_data = data
                        return resp
                except Exception as rep_err:
                    logger.warning(f"[QUERY:{query_id}] JSON repair also failed: {rep_err}")
                
        # Graceful fallback: Extract clean answer text
        is_select_query = any(w in question.lower() for w in ["select", "highlight", "segment", "isolate", "pick"])
        ans_text = raw_text.strip()
        match_ans = re.search(r'"answer"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', raw_text)
        if match_ans:
            try:
                ans_text = match_ans.group(1).encode('utf-8').decode('unicode_escape', errors='ignore')
            except Exception:
                ans_text = match_ans.group(1)
        resp = VisionQueryResponse(
            answer=ans_text,
            intent=IntentType.SELECT_OBJECT if is_select_query else IntentType.VISUAL_QUERY,
            requires_segmentation=is_select_query,
            target=TargetObject(label="object", description=question) if is_select_query else None,
            observations=[],
            possible_explanations=[],
            uncertainty="Model output could not be fully parsed into structured JSON.",
            recommended_next_view=RecommendedNextView(required=False)
        )
        resp._raw_data = {}
        return resp
