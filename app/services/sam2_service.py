"""
SAM 2.1 Neural Segmentation Service Module for GauSSCol Vision.

This module encapsulates Facebook / Meta AI's Segment Anything Model 2.1 (SAM 2.1 Hiera Tiny)
for zero-shot promptable 2D image segmentation in Gaussian Splatting scene exploration.

Architectural Rationale:
SAM 2.1 serves as the crucial 2D grounding bridge between VLM spatial reasoning and 3D Gaussian
selection:
1. The VLM provides multi-view bounding box estimates for an object of interest.
2. SAM 2.1 converts these 2D bounding boxes into precise pixel-accurate binary masks.
3. GauSSCol's WebGPU camera ray intersection projects these masks into 3D Gaussian seeds.

To avoid bloated procedural methods, this module separates concerns into modular classes:
- BoundingBoxPixelConverter: Translates normalized coordinates [0..1] into image pixel space [0..W, 0..H]
  with safety clamping and minimum size guarantees.
- MaskPostProcessor: Evaluates candidate mask IoU scores, selects the top prediction, converts tensors
  to binary NumPy masks, and encodes them as optimized PNG data-URIs.
- SAM2ModelManager: Manages PyTorch device placement ('cuda' vs 'cpu'), HuggingFace transformer
  model/processor loading, warmup, and state tracking.
- SAM2Service: Orchestrates the pipeline and provides singleton access.
"""

import io
import time
import base64
import logging
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
from PIL import Image
import torch

from ..config import settings
from ..schemas.vision import NormalizedBBox

logger = logging.getLogger("gausscol_vision.sam2")


class BoundingBoxPixelConverter:
    """
    Responsible for converting NormalizedBBox coordinates into pixel space.

    Handles edge cases:
    - Normalization clamping to image boundaries [0..width, 0..height].
    - Ensuring bounding boxes possess a minimum dimension (e.g., at least 2 pixels)
      to prevent degenerate prompt embeddings in the SAM 2.1 vision backbone.
    """

    def __init__(self, min_box_size: float = 2.0):
        self.min_box_size = min_box_size

    def convert(self, bbox: NormalizedBBox, width: int, height: int) -> Tuple[float, float, float, float]:
        """
        Converts normalized coordinates to clamped pixel coordinates [x1, y1, x2, y2].
        """
        w, h = float(width), float(height)

        x1 = max(0.0, min(w, bbox.x_min * w))
        y1 = max(0.0, min(h, bbox.y_min * h))
        x2 = max(0.0, min(w, bbox.x_max * w))
        y2 = max(0.0, min(h, bbox.y_max * h))

        # Enforce minimum box dimensions
        if x2 - x1 < self.min_box_size:
            x2 = min(w, x1 + self.min_box_size)
        if y2 - y1 < self.min_box_size:
            y2 = min(h, y1 + self.min_box_size)

        return x1, y1, x2, y2


class MaskPostProcessor:
    """
    Responsible for evaluating SAM 2.1 candidate predictions and serializing masks.

    Why this class exists:
    SAM 2.1 produces 3 candidate masks per prompt to resolve depth ambiguity.
    This processor selects the candidate with the highest predicted IoU score,
    converts it into a binary mask, calculates foreground pixel count, and serializes
    it as an optimized PNG data-URI for zero-copy frontend WebGPU transfer.
    """

    @staticmethod
    def select_best_candidate(
        post_masks: Any,
        iou_scores_tensor: torch.Tensor
    ) -> Tuple[np.ndarray, float, int]:
        """
        Picks the highest scoring mask prediction from SAM's multi-mask output.

        Returns:
            Tuple of (mask_numpy_uint8, best_iou_score, foreground_pixel_count)
        """
        # iou_scores: shape [batch_size, num_boxes, num_candidates] -> [0, 0]
        iou_scores = iou_scores_tensor[0, 0].cpu().numpy()
        best_candidate_idx = int(np.argmax(iou_scores))
        best_iou = float(iou_scores[best_candidate_idx])

        # Extract selected mask [1, 3, H, W] -> [H, W]
        mask_tensor = post_masks[0][0, best_candidate_idx]
        mask_np = mask_tensor.numpy().astype(np.uint8) * 255

        foreground_pixels = int(np.count_nonzero(mask_np))

        return mask_np, best_iou, foreground_pixels

    @staticmethod
    def encode_mask_to_png_data_uri(mask_np: np.ndarray) -> str:
        """
        Encodes a 2D uint8 NumPy mask into a base64 PNG data-URI string.
        """
        mask_pil = Image.fromarray(mask_np, mode="L")
        buf = io.BytesIO()
        mask_pil.save(buf, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


class SAM2ModelManager:
    """
    Manages the lifecycle, device allocation, and initialization of the SAM 2.1 Hiera model.
    """

    def __init__(self, model_id: str, configured_device: str):
        self.model_id = model_id
        self.device = self._resolve_device(configured_device)
        self.model = None
        self.processor = None
        self.ready = False
        self._is_loading = False

    def _resolve_device(self, configured_device: str) -> str:
        """
        Determines the optimal execution device based on hardware availability.
        """
        dev = configured_device.lower()
        if dev == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif dev == "cpu":
            return "cpu"
        # auto fallback
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self) -> bool:
        """
        Loads the HuggingFace Sam2Model and Sam2Processor onto the target device.
        """
        if self.ready:
            return True

        if self._is_loading:
            return False

        self._is_loading = True
        start_time = time.time()
        logger.info("==================================================")
        logger.info(f"Loading SAM 2.1 Hiera Tiny ({self.model_id})...")
        logger.info(f"Device: {self.device.upper()}")
        logger.info("==================================================")

        try:
            from transformers import Sam2Model, Sam2Processor

            self.processor = Sam2Processor.from_pretrained(self.model_id)
            self.model = Sam2Model.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()

            load_ms = (time.time() - start_time) * 1000.0
            self.ready = True
            self._is_loading = False

            logger.info("SAM 2.1 initialized successfully")
            logger.info("Model: Hiera Tiny")
            logger.info(f"Device: {self.device.upper()}")
            logger.info(f"SAM model loaded in {load_ms:.1f}ms.")
            return True
        except Exception as e:
            self._is_loading = False
            self.ready = False
            logger.error(f"Failed to initialize SAM 2.1 model: {e}", exc_info=True)
            return False


class SAM2Service:
    """
    High-level facade for SAM 2.1 neural segmentation in GauSSCol.

    Provides a clean, unified interface for:
    - Loading the model on demand or startup.
    - Segmenting images using normalized bounding boxes.
    - Exposing operational diagnostics and metadata.
    """
    _instance: Optional["SAM2Service"] = None

    def __init__(
        self,
        model_manager: Optional[SAM2ModelManager] = None,
        box_converter: Optional[BoundingBoxPixelConverter] = None,
        post_processor: Optional[MaskPostProcessor] = None
    ):
        self.enabled = settings.sam_enabled
        self.model_manager = model_manager or SAM2ModelManager(
            model_id=settings.sam_model_id,
            configured_device=settings.sam_device
        )
        self.box_converter = box_converter or BoundingBoxPixelConverter()
        self.post_processor = post_processor or MaskPostProcessor()

    @classmethod
    def get_instance(cls) -> "SAM2Service":
        """
        Returns the process-wide singleton instance of the service.
        """
        if cls._instance is None:
            cls._instance = SAM2Service()
        return cls._instance

    # Properties maintaining direct backward-compatibility with existing tests and scripts
    @property
    def model_id(self) -> str:
        return self.model_manager.model_id

    @property
    def device(self) -> str:
        return self.model_manager.device

    @property
    def ready(self) -> bool:
        return self.model_manager.ready

    @ready.setter
    def ready(self, value: bool) -> None:
        self.model_manager.ready = value

    @property
    def model(self):
        return self.model_manager.model

    @model.setter
    def model(self, value):
        self.model_manager.model = value

    @property
    def processor(self):
        return self.model_manager.processor

    @processor.setter
    def processor(self, value):
        self.model_manager.processor = value

    def load_model(self) -> bool:
        """
        Loads the underlying neural model if segmentation is enabled.
        """
        if not self.enabled:
            logger.info("SAM 2.1 is disabled by configuration (SAM_ENABLED=false).")
            return False
        return self.model_manager.load_model()

    def get_info(self) -> Dict[str, Any]:
        """
        Returns runtime status and diagnostic metadata.
        """
        return {
            "enabled": self.enabled,
            "model": "sam2.1_hiera_tiny",
            "model_id": self.model_manager.model_id,
            "device": self.model_manager.device,
            "cuda_available": torch.cuda.is_available(),
            "ready": self.model_manager.ready
        }

    def segment_image(
        self,
        image: Image.Image,
        bbox: NormalizedBBox,
        points: Optional[List[Tuple[float, float, int]]] = None
    ) -> Dict[str, Any]:
        """
        Executes SAM 2.1 segmentation for a single image and normalized bounding box.

        Returns:
            Dictionary matching the GauSSCol SAM API contract:
            - mask_png_base64: Base64 data-URI PNG mask
            - confidence: IoU score of the best mask candidate
            - foreground_pixel_count: Number of non-zero pixels in mask
            - width: Viewport image width
            - height: Viewport image height
            - timing_ms: Total inference latency in milliseconds
        """
        if not self.ready:
            loaded = self.load_model()
            if not loaded:
                raise RuntimeError("SAM 2.1 service is not ready and could not be loaded.")

        t0 = time.time()
        w, h = image.size

        # 1. Convert normalized bounding box to pixel coordinates
        x1, y1, x2, y2 = self.box_converter.convert(bbox, width=w, height=h)

        # 2. Format inputs: SAM 2.1 expects 3-level nesting [[[x1, y1, x2, y2]]]
        inputs = self.processor(
            images=image,
            input_boxes=[[[x1, y1, x2, y2]]],
            return_tensors="pt"
        )

        if self.device != "cpu":
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # 3. Model inference without gradients
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 4. Post-process masks to original image dimensions
        post_masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            binarize=True
        )

        # 5. Select best candidate mask based on IoU score
        mask_np, best_iou, fg_count = self.post_processor.select_best_candidate(
            post_masks=post_masks,
            iou_scores_tensor=outputs.iou_scores
        )

        # 6. Encode mask as PNG data URI
        b64_str = self.post_processor.encode_mask_to_png_data_uri(mask_np)

        elapsed_ms = (time.time() - t0) * 1000.0

        logger.info(
            f"SAM 2.1 segmented view ({w}x{h}): bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], "
            f"iou={best_iou:.3f}, fg_pixels={fg_count}, time={elapsed_ms:.1f}ms"
        )

        return {
            "mask_png_base64": b64_str,
            "confidence": round(best_iou, 4),
            "foreground_pixel_count": fg_count,
            "width": w,
            "height": h,
            "timing_ms": round(elapsed_ms, 2)
        }


def get_sam2_service() -> SAM2Service:
    """
    Convenience factory returning the singleton SAM2Service instance.
    """
    return SAM2Service.get_instance()
