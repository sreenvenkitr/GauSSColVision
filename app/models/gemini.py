import os
import io
import json
import base64
import logging
import requests
from typing import Dict, Any, Optional, List
from PIL import Image

from .base import VisionLanguageModel
from ..schemas.vision import VisionQueryRequest, ViewMetadata
from ..prompts.scene_analysis import SYSTEM_INSTRUCTION, build_user_prompt
from ..config import settings

logger = logging.getLogger("gausscol_vision.gemini")

class GeminiVLM(VisionLanguageModel):
    """
    Google Gemini multimodal Vision-Language Model provider.

    How this class works:
    Translates multi-view camera viewport images and 3D inspection queries into a multi-part
    JSON payload dispatched to Google Generative Language REST APIs.

    Concrete Example:
    -----------------
    When a user asks: "Select the armchair near the desk":
    1. GauSSColUI captures 4 viewpoints: 'current' (user active canvas) and 3 orbital angles.
    2. analyze() serializes each PIL Image into a base64 JPEG data block annotated with its camera ID.
    3. The payload instructs Gemini to return a strict JSON object mapping visible bounding boxes
       across each camera perspective.
    4. _execute_request_with_fallback() attempts 'gemini-3.6-flash' (1.2s response time); if Google Cloud
       returns 404/503/429, it automatically falls back to 'gemini-3.7-flash' or 'gemini-3.5-flash'.
    5. _extract_response_text() filters out internal thinking tokens (reasoning process) and returns
       the clean JSON string for the parser and SAM 2.1 mask generator.
    """

    # Candidate models ordered by latency, quota availability, and capability
    DEFAULT_CANDIDATE_MODELS = [
        "gemini-flash-lite-latest",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-flash-latest"
    ]

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-flash-lite-latest"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = os.getenv("VISION_GEMINI_MODEL", model_name)
        self.ready = bool(self.api_key and self.api_key.strip())
        if not self.ready:
            logger.warning("GeminiVLM initialized without GEMINI_API_KEY.")
        else:
            logger.info(f"GeminiVLM initialized with model: {self.model_name}")

    def get_info(self) -> Dict[str, Any]:
        """Returns provider metadata for health and diagnostics telemetry."""
        return {
            "model": f"Google Gemini ({self.model_name})",
            "device": "cloud_api",
            "cuda_available": False,
            "dtype": "fp16_cloud",
            "ready": bool(self.api_key and self.api_key.strip())
        }

    def analyze(self, request: VisionQueryRequest, images: Dict[str, Image.Image]) -> str:
        """
        Executes multimodal visual reasoning via Google Gemini Cloud REST API.
        
        Execution pipeline:
        1. Validate API credentials.
        2. Serialize multi-view camera images into base64 JPEG parts.
        3. Assemble Gemini REST payload with system instructions and JSON generation config.
        4. Dispatch request to candidate model pool with automatic failover.
        5. Extract and return pure structured JSON response text.
        """
        self._validate_credentials()

        # Step 1: Serialize image views into base64 inline data parts
        image_parts = self._serialize_image_parts(request.views, images)

        # Step 2: Build formatted user query prompt
        user_prompt_text = self._build_user_prompt(request)

        # Step 3: Compose Gemini generateContent payload
        payload = self._compose_request_payload(image_parts, user_prompt_text)

        # Step 4: Dispatch HTTP POST with model fallback pool
        http_response = self._execute_request_with_fallback(payload, num_views=len(images))

        # Step 5: Extract final text content (excluding thought tokens)
        return self._extract_response_text(http_response)

    def _validate_credentials(self) -> None:
        """Ensures a non-empty Gemini API key is configured."""
        if not self.api_key or not self.api_key.strip():
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set your free Gemini API key in the environment or .env file."
            )

    def _serialize_image_parts(
        self,
        views: List[ViewMetadata],
        images: Dict[str, Image.Image]
    ) -> List[Dict[str, Any]]:
        """
        Encodes each PIL Image viewpoint into a compressed base64 JPEG payload block.

        Why JPEG:
        Raw RGBA PNGs from WebGPU can be 2-4MB each. Compressing at quality 85 reduces each image
        to ~40-80KB without degrading Gemini's ability to detect localized object boundaries,
        cutting upload latency from seconds to milliseconds.
        """
        parts = []
        for view in views:
            vid = view.view_id
            if vid in images:
                # Add contextual delimiter header for VLM view tracking
                parts.append({
                    "text": f"--- Camera View ID: {vid} (Role: {view.role}) ---"
                })

                img = images[vid]
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")

                parts.append({
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": b64_data
                    }
                })
        return parts

    def _build_user_prompt(self, request: VisionQueryRequest) -> str:
        """Synthesizes question, view roles, ROI coordinates, and conversation history."""
        return build_user_prompt(
            question=request.question,
            query_scope=request.query_scope.value,
            views_meta=[v.model_dump() if hasattr(v, "model_dump") else v.dict() for v in request.views],
            roi_box=(request.roi.model_dump() if hasattr(request.roi, "model_dump") else request.roi.dict()) if request.roi else None,
            conversation_history=[t.model_dump() if hasattr(t, "model_dump") else t.dict() for t in request.conversation_history]
        )

    def _compose_request_payload(
        self,
        image_parts: List[Dict[str, Any]],
        user_prompt_text: str
    ) -> Dict[str, Any]:
        """
        Assembles the Gemini REST generateContent request body with system instructions,
        JSON MIME type enforcement, and thinking budget controls.
        """
        contents_parts = list(image_parts)
        contents_parts.append({"text": user_prompt_text})

        return {
            "system_instruction": {
                "parts": [{"text": SYSTEM_INSTRUCTION}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": contents_parts
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

    def _get_candidate_models(self) -> List[str]:
        """Returns unique ordered candidate model names with the active model prioritized."""
        candidates = [self.model_name] + self.DEFAULT_CANDIDATE_MODELS
        unique_models = []
        for m in candidates:
            if m not in unique_models:
                unique_models.append(m)
        return unique_models

    def _execute_request_with_fallback(self, payload: Dict[str, Any], num_views: int) -> requests.Response:
        """
        Dispatches HTTP POST to Gemini REST endpoint, iterating through candidate models on failure.

        Why fallback exists:
        Google frequently releases newer flash checkpoints or deprecates preview names.
        If the primary model returns 404 (model not found) or 429 (quota exhausted),
        the loop automatically cascades to alternative compatible endpoints without crashing the user session.
        """
        headers = {"Content-Type": "application/json"}
        candidate_models = self._get_candidate_models()

        resp = None
        last_error = ""

        for model_id in candidate_models:
            target_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={self.api_key.strip()}"
            logger.info(f"Dispatching query to Gemini API ({model_id}, {num_views} views)...")
            try:
                r = requests.post(target_url, headers=headers, json=payload, timeout=45)
                if r.status_code == 200:
                    resp = r
                    self.model_name = model_id  # Retain the verified working model for subsequent requests
                    break
                else:
                    last_error = f"Gemini API error on {model_id} ({r.status_code}): {r.text}"
                    logger.warning(last_error)
            except Exception as e:
                last_error = f"Network error connecting to Gemini on {model_id}: {e}"
                logger.warning(last_error)

        if resp is None:
            raise RuntimeError(last_error or "All Gemini models in pool failed.")

        return resp

    def _extract_response_text(self, resp: requests.Response) -> str:
        """
        Validates API response payload and extracts clean text content.
        Filters out internal thinking tokens so only the raw JSON payload is parsed.
        """
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

        # Accumulate all text parts, explicitly ignoring thought/reasoning blocks
        text_parts = []
        for p in cand_parts:
            if "text" in p and not p.get("thought", False):
                text_parts.append(p["text"])

        return "".join(text_parts) if text_parts else cand_parts[0].get("text", "")


class MockVisionModel(VisionLanguageModel):
    """
    Offline simulation model for unit testing and local development without cloud API credentials.

    Why hardcoded keywords like ["chair", "armchair", "seat"] exist in this class:
    -------------------------------------------------------------------------------
    This class is NOT used during live production with Gemini. It is exclusively an offline
    mock simulator activated when:
    1. Running pytest unit tests (`tests/test_api.py`, `tests/test_sam.py`), or
    2. `VISION_USE_MOCK_MODEL=true` in `.env`, or
    3. No `GEMINI_API_KEY` is present in the local environment.

    To verify the downstream SAM 2.1 segmentation and FastAPI routing without calling Google's
    paid cloud API or requiring network access, this mock inspects the test query string:
    - If the user commands: "Select the brown chair":
      The mock matches "select" + "chair", immediately returning a synthetic 2D bounding box
      [x_min: 0.25, y_min: 0.30, x_max: 0.65, y_max: 0.85] for target "chair".
      This enables SAM 2.1 to generate real pixel masks and test the entire WebGPU pipeline
      deterministically.
    - If the user asks an informational question: "Where is the armchair?":
      The mock matches "armchair", returning descriptive observations without triggering segmentation.
    - Any other query:
      Returns a generic structural observation.
    """

    def __init__(self):
        self.ready = True
        logger.info("Initialized MockVisionModel (Offline Test Mode)")

    def get_info(self) -> Dict[str, Any]:
        """Returns mock metadata confirming offline execution."""
        return {
            "model": "MockVisionModel (Offline Test Mode)",
            "device": "cpu",
            "cuda_available": False,
            "dtype": "float32",
            "ready": True
        }

    def analyze(self, request: VisionQueryRequest, images: Dict[str, Image.Image]) -> str:
        """
        Simulates structured VLM JSON responses based on keyword inspection for offline testing.
        """
        q = request.question.lower().strip()
        scope = request.query_scope.value
        view_ids = list(images.keys()) or ["current"]
        primary_view = view_ids[0] if view_ids else "current"

        # Case 1: 2D ROI selection scope simulation
        if scope == "roi_2d":
            return self._mock_roi_response(view_ids, primary_view)

        # Case 2: Object selection commands ("Select the brown chair", "Segment the seat", etc.)
        if any(w in q for w in ["select", "highlight", "segment", "isolate", "pick"]):
            return self._mock_select_response(q, view_ids, primary_view)

        # Case 3: Visual query about chair / armchair (e.g., test_api.py: "Where is the chair?")
        if any(w in q for w in ["chair", "armchair", "seat"]):
            return self._mock_chair_query_response(primary_view)

        # Case 4: Generic scene inspection fallback
        return self._mock_generic_response(request.question, primary_view)

    def _mock_roi_response(self, view_ids: List[str], primary_view: str) -> str:
        """Simulates inspection response for a user-drawn 2D rectangle."""
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

    def _mock_select_response(self, question: str, view_ids: List[str], primary_view: str) -> str:
        """
        Simulates multi-view grounding for selection commands.
        Extracts target label (defaults to 'chair' if chair keywords are present, or 'object').
        """
        target_label = "chair" if any(w in question for w in ["chair", "armchair", "seat"]) else "object"
        target_desc = f"selected {target_label}"

        # Generate synthetic grounded view bounding boxes for up to 4 viewpoints
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

    def _mock_chair_query_response(self, primary_view: str) -> str:
        """Simulates informational query response for test_api.py."""
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

    def _mock_generic_response(self, question: str, primary_view: str) -> str:
        """Simulates general visual query inspection without segmentation."""
        resp = {
            "answer": f"Based on rendered viewpoint(s), the visual inspection corresponds to query: '{question}'.",
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


_model_instance = None

def get_vlm_model() -> VisionLanguageModel:
    """
    Returns the active VisionLanguageModel provider instance.
    Supports 'gemini' (default) and OpenAI-compatible providers like 'openrouter'.
    Falls back to MockVisionModel for offline testing or when keys are absent.
    """
    global _model_instance
    if settings.use_mock_model:
        return MockVisionModel()
    if _model_instance is None:
        provider = (settings.vision_provider or "gemini").lower()
        if provider in ("openrouter", "openai", "openai_compatible"):
            from .openai_compatible import OpenAICompatibleVLM
            logger.info(f"Initializing OpenAICompatibleVLM provider for {provider}...")
            _model_instance = OpenAICompatibleVLM(
                api_key=settings.openrouter_api_key,
                model_name=settings.openrouter_model,
                base_url=settings.openrouter_base_url
            )
        elif settings.gemini_api_key and settings.gemini_api_key.strip():
            logger.info("Initializing GeminiVLM provider...")
            _model_instance = GeminiVLM(settings.gemini_api_key.strip(), settings.gemini_model)
        else:
            logger.warning("No API credentials detected for active provider. Using MockVisionModel fallback.")
            _model_instance = MockVisionModel()
    return _model_instance
