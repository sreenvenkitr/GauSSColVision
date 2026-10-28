"""
Structured Response Parser Module for GauSSCol Vision.

This module is responsible for translating raw text produced by Vision-Language Models (VLMs)
into strongly-typed Pydantic domain models (`VisionQueryResponse`, `VisionObservation`, `TargetObject`).

Architectural Rationale:
Generative language models frequently exhibit formatting variances:
1. Markdown framing (e.g., ```json ... ``` blocks).
2. Subtle syntax corruptions (trailing commas, unbalanced braces or unescaped quotes).
3. Ambiguous coordinate systems (e.g., Google/Gemini [ymin, xmin, ymax, xmax] lists,
   0..1000 integer ranges, or Web-standard [x_min, y_min, x_max, y_max] dictionaries).

To maintain clean separation of concerns and avoid bloated monolithic parsing routines,
this module decomposes parsing into distinct single-responsibility classes:
- BoundingBoxParser: Normalizes coordinate representations into [0.0, 1.0] NormalizedBBox.
- JsonSanitizer: Extracts JSON substrings and applies heuristic repairs.
- TargetObjectExtractor: Parses and cleans object descriptions, applying prompt heuristics.
- ObservationParser: Maps detected features and bounding boxes to VisionObservation instances.
- StructuredResponseParser: High-level orchestrator coordinating the sub-components.
"""

import re
import json
import logging
from typing import Any, Optional, List, Dict

from ..schemas.vision import (
    VisionQueryResponse,
    VisionObservation,
    RecommendedNextView,
    NormalizedBBox,
    IntentType,
    TargetObject
)

logger = logging.getLogger("gausscol_vision.response_parser")


class BoundingBoxParser:
    """
    Responsible solely for converting various bounding box data structures into a NormalizedBBox.

    Supported Input Formats:
    1. Already instantiated `NormalizedBBox`.
    2. Dictionaries with keys ('x_min', 'y_min', 'x_max', 'y_max') or ('xmin', 'ymin', 'xmax', 'ymax').
    3. Dictionaries containing a Google-style 'box_2d': [ymin, xmin, ymax, xmax] entry.
    4. Sequences [ymin, xmin, ymax, xmax] in either 0.0..1.0 float format or 0..1000 integer range.
    """

    @staticmethod
    def parse(bbox_data: Any) -> Optional[NormalizedBBox]:
        """
        Extracts and normalizes bounding box coordinates to [0.0, 1.0].
        Returns None if data is missing, invalid, or cannot be parsed.
        """
        if not bbox_data:
            return None

        if isinstance(bbox_data, NormalizedBBox):
            return bbox_data

        if isinstance(bbox_data, dict):
            return BoundingBoxParser._parse_dict(bbox_data)

        if isinstance(bbox_data, (list, tuple)) and len(bbox_data) >= 4:
            return BoundingBoxParser._parse_sequence(bbox_data)

        return None

    @staticmethod
    def _parse_dict(data: Dict[str, Any]) -> Optional[NormalizedBBox]:
        """
        Parses bounding box representations stored in dictionary mappings.
        """
        # Format: Google Gemini 'box_2d' nested key
        if "box_2d" in data:
            return BoundingBoxParser.parse(data["box_2d"])

        # Format: Named coordinates
        x_min = data.get("x_min", data.get("xmin"))
        y_min = data.get("y_min", data.get("ymin"))
        x_max = data.get("x_max", data.get("xmax"))
        y_max = data.get("y_max", data.get("ymax"))

        if None in (x_min, y_min, x_max, y_max):
            return None

        try:
            fx_min, fy_min = float(x_min), float(y_min)
            fx_max, fy_max = float(x_max), float(y_max)
            return BoundingBoxParser._normalize_and_clamp(fx_min, fy_min, fx_max, fy_max)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_sequence(coords: Any) -> Optional[NormalizedBBox]:
        """
        Parses 4-element coordinate sequences [ymin, xmin, ymax, xmax] (Gemini standard).
        """
        try:
            c1, c2, c3, c4 = [float(v) for v in coords[:4]]
            # Gemini returns [ymin, xmin, ymax, xmax]
            y_min, x_min, y_max, x_max = c1, c2, c3, c4
            return BoundingBoxParser._normalize_and_clamp(x_min, y_min, x_max, y_max)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _normalize_and_clamp(x1: float, y1: float, x2: float, y2: float) -> NormalizedBBox:
        """
        Scales 0..1000 coordinate systems down to 0..1 and clamps values to [0.0, 1.0].
        """
        # Auto-detect 0..1000 integer ranges produced by some VLM prompt variants
        if max(x1, y1, x2, y2) > 1.0:
            x1 /= 1000.0
            y1 /= 1000.0
            x2 /= 1000.0
            y2 /= 1000.0

        x_min = max(0.0, min(1.0, min(x1, x2)))
        y_min = max(0.0, min(1.0, min(y1, y2)))
        x_max = max(0.0, min(1.0, max(x1, x2)))
        y_max = max(0.0, min(1.0, max(y1, y2)))

        return NormalizedBBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max
        )


class JsonSanitizer:
    """
    Responsible for isolating JSON payloads from markdown text and applying syntax corrections.
    """

    @staticmethod
    def extract_json_candidate(text: str) -> Optional[str]:
        """
        Strips markdown code fences and extracts the candidate JSON string bounded by { and }.
        """
        cleaned = text.strip()

        # Check for ```json ... ``` code blocks
        if "```" in cleaned:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()
            else:
                cleaned = re.sub(r"```[a-zA-Z]*", "", cleaned).replace("```", "").strip()

        # Locate the outermost curly braces
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return cleaned[start_idx : end_idx + 1]

        return None

    @staticmethod
    def repair_syntax(candidate: str) -> str:
        """
        Applies heuristic string repairs for common VLM JSON generation anomalies:
        - Trailing commas before closing braces/brackets
        - Unclosed quotation marks
        - Missing trailing closing braces/brackets
        """
        repaired = candidate

        # 1. Remove trailing commas: e.g. ", }" or ", ]"
        repaired = re.sub(r",\s*([\]}])", r"\1", repaired)

        # 2. Fix unbalanced double quotes (append closing quote if odd count)
        if repaired.count('"') % 2 != 0:
            repaired += '"'

        # 3. Balance unclosed curly braces
        open_braces = repaired.count("{") - repaired.count("}")
        if open_braces > 0:
            repaired += "}" * open_braces

        # 4. Balance unclosed square brackets
        open_brackets = repaired.count("[") - repaired.count("]")
        if open_brackets > 0:
            repaired += "]" * open_brackets

        return repaired


class TargetObjectExtractor:
    """
    Responsible for extracting and resolving the TargetObject for 3D selection.

    Why this class exists:
    VLMs sometimes produce vague labels like 'object' or 'target' even when the user specifically
    asked to 'select the brown armchair'. This class inspects both the model JSON and the user's
    original prompt to produce a concrete, user-friendly object label.
    """

    @staticmethod
    def extract(target_data: Any, intent: IntentType, question: str) -> Optional[TargetObject]:
        """
        Resolves target object label and description from structured data or prompt text.
        """
        # If model provided structured target object dictionary
        if target_data and isinstance(target_data, dict):
            label = str(target_data.get("label", "object")).strip()
            description = str(target_data.get("description", label)).strip()

            # If the label is a generic placeholder and the user wants to select an object, extract from question
            if label.lower() in ("object", "target", "target object", "") and intent == IntentType.SELECT_OBJECT:
                inferred_label = TargetObjectExtractor._clean_question_for_target(question)
                if inferred_label:
                    label = inferred_label
                    description = inferred_label

            return TargetObject(label=label, description=description)

        # Fallback when intent is SELECT_OBJECT but no target dictionary was provided
        if intent == IntentType.SELECT_OBJECT:
            inferred_label = TargetObjectExtractor._clean_question_for_target(question)
            return TargetObject(
                label=inferred_label or "object",
                description=question
            )

        return None

    @staticmethod
    def _clean_question_for_target(question: str) -> str:
        """
        Strips common command verbs and prepositional phrases from the question.
        Example: "select the red chair in the scene" -> "red chair"
        """
        cleaned = re.sub(
            r"^(select|highlight|segment|isolate|pick|find)\s+(the\s+)?",
            "",
            question,
            flags=re.I
        ).strip()
        cleaned = re.sub(
            r"\s+(in\s+the\s+(scene|room|view)|from\s+the\s+scene).*$",
            "",
            cleaned,
            flags=re.I
        ).strip()
        return cleaned


class ObservationParser:
    """
    Responsible for converting raw observation dictionaries into typed VisionObservation objects.
    """

    def __init__(self, bbox_parser: Optional[BoundingBoxParser] = None):
        self.bbox_parser = bbox_parser or BoundingBoxParser()

    def parse_observations(self, raw_items: Any, query_id: str) -> List[VisionObservation]:
        """
        Parses a list of raw observation dictionaries into VisionObservation instances.
        """
        if not isinstance(raw_items, list):
            return []

        observations: List[VisionObservation] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue

            try:
                raw_bbox = item.get("bbox") or item.get("box_2d")
                bbox_obj = self.bbox_parser.parse(raw_bbox)

                conf_val = item.get("confidence")
                conf_float = float(conf_val) if conf_val is not None else None

                obs = VisionObservation(
                    label=str(item.get("label", "detected feature")),
                    view_id=str(item.get("view_id", "current")),
                    bbox=bbox_obj,
                    confidence=conf_float,
                    description=item.get("description")
                )
                observations.append(obs)
            except Exception as e:
                logger.warning(f"[QUERY:{query_id}] Error parsing observation item: {e}")

        return observations


class StructuredResponseParser:
    """
    High-level orchestrator for parsing multimodal VLM text into typed VisionQueryResponse models.

    Responsibilities:
    - Coordinates JsonSanitizer, TargetObjectExtractor, ObservationParser, and BoundingBoxParser.
    - Decodes and repairs JSON outputs.
    - Provides backward-compatible entry points: `parse_response` and `extract_bounding_box`.
    - Generates reliable fallback responses when VLM output fails to produce valid JSON.
    """

    def __init__(
        self,
        bbox_parser: Optional[BoundingBoxParser] = None,
        json_sanitizer: Optional[JsonSanitizer] = None,
        target_extractor: Optional[TargetObjectExtractor] = None,
        observation_parser: Optional[ObservationParser] = None
    ):
        self.bbox_parser = bbox_parser or BoundingBoxParser()
        self.json_sanitizer = json_sanitizer or JsonSanitizer()
        self.target_extractor = target_extractor or TargetObjectExtractor()
        self.observation_parser = observation_parser or ObservationParser(self.bbox_parser)

    def extract_bounding_box(self, bbox_data: Any) -> Optional[NormalizedBBox]:
        """
        Normalizes bounding box coordinates across varied input representations.
        Preserves backward compatibility for callers and test suites.
        """
        return self.bbox_parser.parse(bbox_data)

    def parse_response(self, raw_text: str, query_id: str, question: str = "") -> VisionQueryResponse:
        """
        Parses raw text from the VLM, handling formatting variances, repairing JSON if needed,
        and constructing a typed VisionQueryResponse. Attaches the raw dictionary to `_raw_data`.
        """
        if not raw_text or not raw_text.strip():
            logger.warning(f"[QUERY:{query_id}] Model returned empty string.")
            return self._build_empty_response()

        # Step 1: Extract candidate JSON substring
        candidate = self.json_sanitizer.extract_json_candidate(raw_text)

        if candidate:
            # Step 2: Attempt standard JSON parse
            try:
                data = json.loads(candidate)
                return self._build_typed_response(data, raw_text, question, query_id)
            except Exception as json_err:
                logger.warning(f"[QUERY:{query_id}] Initial JSON parse failed ({json_err}). Attempting repair...")

            # Step 3: Attempt heuristic JSON repair
            repaired = self.json_sanitizer.repair_syntax(candidate)
            try:
                data = json.loads(repaired)
                logger.info(f"[QUERY:{query_id}] JSON successfully repaired and decoded.")
                return self._build_typed_response(data, raw_text, question, query_id)
            except Exception as rep_err:
                logger.warning(f"[QUERY:{query_id}] JSON repair also failed ({rep_err}).")

        # Step 4: Graceful fallback response when JSON cannot be parsed
        return self._build_fallback_response(raw_text, question)

    def _build_typed_response(
        self,
        data: Dict[str, Any],
        fallback_text: str,
        question: str,
        query_id: str
    ) -> VisionQueryResponse:
        """
        Builds a fully-validated VisionQueryResponse from a successfully parsed dictionary.
        """
        # Determine intent: check both explicit field and prompt intent keywords
        raw_intent = str(data.get("intent", "")).upper()
        is_select_query = any(w in question.lower() for w in ["select", "highlight", "segment", "isolate", "pick"])
        if raw_intent == "SELECT_OBJECT" or is_select_query:
            intent = IntentType.SELECT_OBJECT
            requires_segmentation = True
        else:
            intent = IntentType.VISUAL_QUERY
            requires_segmentation = False

        # Extract target object
        target_obj = self.target_extractor.extract(data.get("target"), intent, question)

        # Parse observations
        observations = self.observation_parser.parse_observations(data.get("observations", []), query_id)

        # Parse recommended next view
        rec_data = data.get("recommended_next_view", {})
        rec_view = RecommendedNextView(
            required=bool(rec_data.get("required", False)),
            reason=rec_data.get("reason"),
            yaw_delta_deg=rec_data.get("yaw_delta_deg"),
            pitch_delta_deg=rec_data.get("pitch_delta_deg"),
            zoom_factor=rec_data.get("zoom_factor")
        )

        resp = VisionQueryResponse(
            answer=str(data.get("answer", fallback_text.strip())),
            intent=intent,
            requires_segmentation=requires_segmentation,
            target=target_obj,
            observations=observations,
            possible_explanations=[str(e) for e in data.get("possible_explanations", [])],
            uncertainty=data.get("uncertainty"),
            recommended_next_view=rec_view
        )
        resp._raw_data = data
        return resp

    def _build_empty_response(self) -> VisionQueryResponse:
        """
        Builds response for an empty model output.
        """
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

    def _build_fallback_response(self, raw_text: str, question: str) -> VisionQueryResponse:
        """
        Builds a fallback response extracting any recoverable answer text from unstructured output.
        """
        is_select_query = any(w in question.lower() for w in ["select", "highlight", "segment", "isolate", "pick"])
        ans_text = raw_text.strip()

        # Heuristic: try regex capture of "answer": "..." field if partial JSON exists
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
