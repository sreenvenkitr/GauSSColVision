import os
import io
import json
import base64
import logging
import re
import requests
from typing import Dict, Any, Optional, List
from PIL import Image

from .base import VisionLanguageModel
from ..schemas.vision import VisionQueryRequest, ViewMetadata
from ..prompts.scene_analysis import SYSTEM_INSTRUCTION, build_user_prompt
from ..config import settings

logger = logging.getLogger("gausscol_vision.openai_compatible")


class OpenAICompatibleVLM(VisionLanguageModel):
    """
    OpenAI-compatible Multimodal Vision-Language Model provider.

    Supports OpenRouter, Groq Cloud, local Ollama (v1/chat/completions),
    vLLM, or any OpenAI-compliant multimodal endpoint.

    How this class works:
    1. Encodes multi-view camera viewport images into base64 JPEG data URLs.
    2. Packages system prompt and image contents into OpenAI /chat/completions format.
    3. Issues HTTP POST request to the configured base URL (default: OpenRouter).
    4. Extracts response text, stripping thinking tokens if present, returning structured JSON.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.api_key = (api_key or settings.openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")).strip()
        self.model_name = (model_name or settings.openrouter_model or "nvidia/nemotron-3.5-super-vl:free").strip()
        self.base_url = (base_url or settings.openrouter_base_url or "https://openrouter.ai/api/v1").rstrip("/")
        self.ready = bool(self.api_key or "localhost" in self.base_url or "127.0.0.1" in self.base_url)

        if not self.ready:
            logger.warning("OpenAICompatibleVLM initialized without API key (required for cloud endpoints).")
        else:
            logger.info(f"OpenAICompatibleVLM initialized (model: {self.model_name}, endpoint: {self.base_url})")

    def get_info(self) -> Dict[str, Any]:
        """Returns provider metadata for health and diagnostics telemetry."""
        return {
            "model": f"OpenAI-Compatible ({self.model_name})",
            "device": "cloud_or_local_api",
            "cuda_available": False,
            "dtype": "fp16_api",
            "base_url": self.base_url,
            "ready": self.ready
        }

    def analyze(self, request: VisionQueryRequest, images: Dict[str, Image.Image]) -> str:
        """
        Executes multimodal visual reasoning via OpenAI-compatible REST endpoint.
        """
        self._validate_credentials()

        # Step 1: Format multi-view image parts into OpenAI content objects
        content_parts = self._serialize_content_parts(request.views, images, request)

        # Step 2: Build messages payload
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": content_parts}
        ]

        # Step 3: Dispatch request
        response_text = self._execute_chat_completion(messages)
        return response_text

    def _validate_credentials(self) -> None:
        """Ensures credentials or local endpoint availability."""
        is_local = "localhost" in self.base_url or "127.0.0.1" in self.base_url
        if not is_local and not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY (or OPENAI_API_KEY) is not set. "
                "Please configure it in .env or switch VISION_PROVIDER to 'gemini'."
            )

    def _serialize_content_parts(
        self,
        views: List[ViewMetadata],
        images: Dict[str, Image.Image],
        request: VisionQueryRequest
    ) -> List[Dict[str, Any]]:
        """
        Builds the multimodal user content array containing image URLs and query text.
        """
        parts: List[Dict[str, Any]] = []

        for view in views:
            vid = view.view_id
            if vid in images:
                parts.append({
                    "type": "text",
                    "text": f"--- Camera View ID: {vid} (Role: {view.role}) ---"
                })

                img = images[vid]
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")

                parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_data}"
                    }
                })

        user_prompt_text = build_user_prompt(
            question=request.question,
            query_scope=request.query_scope.value,
            views_meta=[v.model_dump() if hasattr(v, "model_dump") else v.dict() for v in request.views],
            roi_box=(request.roi.model_dump() if hasattr(request.roi, "model_dump") else request.roi.dict()) if request.roi else None,
            conversation_history=[t.model_dump() if hasattr(t, "model_dump") else t.dict() for t in request.conversation_history]
        )

        parts.append({
            "type": "text",
            "text": user_prompt_text
        })

        return parts

    def _execute_chat_completion(self, messages: List[Dict[str, Any]]) -> str:
        """
        Sends HTTP POST to /chat/completions endpoint.
        Tries with response_format: json_object first, falling back without if unsupported.
        """
        target_url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/GauSSCol",
            "X-Title": "GauSSCol 3D Vision"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        base_payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 4096
        }

        # Attempt 1: With JSON response format enforcement
        payload_with_json = dict(base_payload)
        payload_with_json["response_format"] = {"type": "json_object"}

        try:
            logger.info(f"Dispatching visual query to OpenAI-compatible provider ({self.model_name})...")
            r = requests.post(target_url, headers=headers, json=payload_with_json, timeout=60)
            if r.status_code == 200:
                return self._extract_content_from_response(r.json())
            elif r.status_code == 400 and "response_format" in r.text.lower():
                logger.info("Provider rejected response_format parameter. Retrying without it...")
                r_fallback = requests.post(target_url, headers=headers, json=base_payload, timeout=60)
                if r_fallback.status_code == 200:
                    return self._extract_content_from_response(r_fallback.json())
                else:
                    raise RuntimeError(f"API error ({r_fallback.status_code}): {r_fallback.text}")
            else:
                raise RuntimeError(f"API error ({r.status_code}): {r.text}")
        except requests.RequestException as e:
            raise RuntimeError(f"Network error connecting to {target_url}: {e}")

    def _extract_content_from_response(self, data: Dict[str, Any]) -> str:
        """
        Extracts raw response text from OpenAI chat completion payload and cleans thinking blocks.
        """
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"No choices returned from chat completion API: {data}")

        message = choices[0].get("message", {})
        content = message.get("content", "")

        if not content and "text" in choices[0]:
            content = choices[0]["text"]

        # Filter out <think>...</think> tags if model produces chain-of-thought tokens
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content
