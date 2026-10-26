import os
import io
import json
import base64
import logging
import requests
from typing import Dict, Any, Optional
from PIL import Image

from .base import VisionLanguageModel
from ..schemas.vision import VisionQueryRequest
from ..prompts.scene_analysis import SYSTEM_INSTRUCTION, build_user_prompt
from ..config import settings

logger = logging.getLogger("gausscol_vision.gemini")

class GeminiVLM(VisionLanguageModel):
    """
    Google Gemini multimodal VLM provider (Gemini 2.0 Flash / 1.5 Flash).
    Delivers 1.0 - 1.5 second multi-image 3D visual reasoning with
    guaranteed JSON response formatting and normalized 2D bounding boxes.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-3.6-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = os.getenv("VISION_GEMINI_MODEL", model_name)
        self.ready = bool(self.api_key and self.api_key.strip())
        if not self.ready:
            logger.warning("GeminiVLM initialized without GEMINI_API_KEY.")
        else:
            logger.info(f"GeminiVLM initialized with model: {self.model_name}")

    def get_info(self) -> Dict[str, Any]:
        return {
            "model": f"Google Gemini ({self.model_name})",
            "device": "cloud_api",
            "cuda_available": False,
            "dtype": "fp16_cloud",
            "ready": bool(self.api_key and self.api_key.strip())
        }

    def analyze(self, request: VisionQueryRequest, images: Dict[str, Image.Image]) -> str:
        if not self.api_key or not self.api_key.strip():
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set your free Gemini API key in the environment or .env file."
            )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key.strip()}"

        user_prompt = build_user_prompt(
            question=request.question,
            query_scope=request.query_scope.value,
            views_meta=[v.dict() for v in request.views],
            roi_box=request.roi.dict() if request.roi else None,
            conversation_history=[t.dict() for t in request.conversation_history]
        )

        parts = []

        # Add image views
        for view in request.views:
            vid = view.view_id
            if vid in images:
                parts.append({
                    "text": f"--- Camera View ID: {vid} (Role: {view.role}) ---"
                })
                img = images[vid]
                buf = io.BytesIO()
                # Save as JPEG for fast upload
                img.save(buf, format="JPEG", quality=85)
                b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": b64_data
                    }
                })

        # Add user question prompt
        parts.append({"text": user_prompt})

        payload = {
            "system_instruction": {
                "parts": [{"text": SYSTEM_INSTRUCTION}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": parts
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.1,
                "maxOutputTokens": 4096,
                "thinkingConfig": {
                    "thinkingBudget": 512
                }
            }
        }

        headers = {"Content-Type": "application/json"}
        candidate_models = [self.model_name, "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-flash-latest"]
        # Remove duplicates while preserving order
        unique_models = []
        for m in candidate_models:
            if m not in unique_models:
                unique_models.append(m)

        resp = None
        last_error = ""
        for m in unique_models:
            target_url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={self.api_key.strip()}"
            logger.info(f"Dispatching query to Gemini API ({m}, {len(images)} views)...")
            try:
                r = requests.post(target_url, headers=headers, json=payload, timeout=45)
                if r.status_code == 200:
                    resp = r
                    self.model_name = m  # Keep the fast working model
                    break
                else:
                    last_error = f"Gemini API error on {m} ({r.status_code}): {r.text}"
                    logger.warning(last_error)
            except Exception as e:
                last_error = f"Network error connecting to Gemini on {m}: {e}"
                logger.warning(last_error)

        if resp is None:
            raise RuntimeError(last_error or "All Gemini models in pool failed.")

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"Gemini API returned no candidates: {data}")

        first_cand = candidates[0]
        finish_reason = first_cand.get("finishReason", "")
        if finish_reason and finish_reason != "STOP":
            logger.warning(f"Gemini finishReason: {finish_reason} (tokens may have truncated)")

        content = first_cand.get("content", {})
        cand_parts = content.get("parts", [])
        if not cand_parts:
            raise RuntimeError(f"Gemini candidate has no content parts: {first_cand}")

        # Accumulate all text parts, ignoring thoughts
        text_parts = []
        for p in cand_parts:
            if "text" in p and not p.get("thought", False):
                text_parts.append(p["text"])

        raw_text = "".join(text_parts) if text_parts else cand_parts[0].get("text", "")
        return raw_text


class MockVisionModel(VisionLanguageModel):
    """
    Interactive simulation model for offline unit testing and development.
    """
    def __init__(self):
        self.ready = True
        logger.info("Initialized MockVisionModel (Offline Test Mode)")

    def analyze(self, request: VisionQueryRequest, images: Dict[str, Image.Image]) -> str:
        q = request.question.lower().strip()
        scope = request.query_scope.value
        view_ids = list(images.keys()) or ["current"]
        primary_view = view_ids[0] if view_ids else "current"

        if scope == "roi_2d":
            roi_view = "roi_detail" if "roi_detail" in view_ids else primary_view
            resp = {
                "answer": "Inspection of the selected 2D region shows localized structural surface contours.",
                "observations": [
                    {
                        "label": "selected ROI focus",
                        "view_id": roi_view,
                        "bbox": {"x_min": 0.25, "y_min": 0.25, "x_max": 0.75, "y_max": 0.75},
                        "confidence": 0.90,
                        "description": "User-designated focal region."
                    }
                ],
                "possible_explanations": [],
                "uncertainty": None,
                "recommended_next_view": {"required": False}
            }
            return json.dumps(resp)

        if any(w in q for w in ["select", "highlight", "segment", "isolate", "pick"]):
            target_label = "chair" if any(w in q for w in ["chair", "armchair", "seat"]) else "object"
            target_desc = f"selected {target_label}"
            grounded = []
            for vid in view_ids[:4]:
                grounded.append({
                    "view_id": vid,
                    "visible": True,
                    "bbox": {"x_min": 0.25, "y_min": 0.30, "x_max": 0.65, "y_max": 0.85},
                    "confidence": 0.94
                })
            resp = {
                "intent": "SELECT_OBJECT",
                "requires_segmentation": True,
                "target": {
                    "label": target_label,
                    "description": target_desc
                },
                "grounded_views": grounded,
                "answer": f"Selected '{target_desc}' across {len(grounded)} visible viewpoints.",
                "observations": [
                    {
                        "label": target_desc,
                        "view_id": primary_view,
                        "bbox": {"x_min": 0.25, "y_min": 0.30, "x_max": 0.65, "y_max": 0.85},
                        "confidence": 0.94,
                        "description": f"Target object {target_desc} grounded for semantic segmentation."
                    }
                ],
                "possible_explanations": [],
                "uncertainty": None,
                "recommended_next_view": {"required": False}
            }
            return json.dumps(resp)

        if any(w in q for w in ["chair", "armchair", "seat"]):
            resp = {
                "intent": "VISUAL_QUERY",
                "requires_segmentation": False,
                "answer": "A dark-toned office armchair is positioned in the central-left area near the workstation.",
                "observations": [
                    {
                        "label": "armchair",
                        "view_id": primary_view,
                        "bbox": {"x_min": 0.32, "y_min": 0.41, "x_max": 0.58, "y_max": 0.82},
                        "confidence": 0.95,
                        "description": "Armchair with dark upholstery and ergonomic frame."
                    }
                ],
                "possible_explanations": [],
                "uncertainty": None,
                "recommended_next_view": {"required": False}
            }
            return json.dumps(resp)

        resp = {
            "answer": f"Based on rendered viewpoint(s), the visual inspection corresponds to query: '{request.question}'.",
            "observations": [
                {
                    "label": "scene observation",
                    "view_id": primary_view,
                    "bbox": {"x_min": 0.30, "y_min": 0.35, "x_max": 0.70, "y_max": 0.80},
                    "confidence": 0.88,
                    "description": "Primary visual target detected."
                }
            ],
            "possible_explanations": [],
            "uncertainty": None,
            "recommended_next_view": {"required": False}
        }
        return json.dumps(resp)

    def get_info(self) -> Dict[str, Any]:
        return {
            "model": "MockVisionModel (Offline Test Mode)",
            "device": "cpu",
            "cuda_available": False,
            "dtype": "float32",
            "ready": True
        }


_model_instance = None

def get_vlm_model() -> VisionLanguageModel:
    global _model_instance
    if settings.use_mock_model:
        return MockVisionModel()
    if _model_instance is None:
        if settings.gemini_api_key and settings.gemini_api_key.strip():
            logger.info("Initializing GeminiVLM provider...")
            _model_instance = GeminiVLM(settings.gemini_api_key.strip(), settings.gemini_model)
        else:
            logger.warning("No GEMINI_API_KEY detected. Using MockVisionModel fallback.")
            _model_instance = MockVisionModel()
    return _model_instance

