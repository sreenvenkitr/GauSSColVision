import os
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

class SAM2Service:
    """
    SAM 2.1 Hiera Tiny segmentation service.
    Accepts an image and 2D bounding box (normalized or pixel),
    runs SAM 2.1 neural segmentation, and produces an optimized
    binary PNG mask encoded as a base64 data URI.
    """
    _instance: Optional["SAM2Service"] = None

    def __init__(self):
        self.model_id = settings.sam_model_id
        self.enabled = settings.sam_enabled
        self.device = self._resolve_device()
        self.model = None
        self.processor = None
        self.ready = False
        self._is_loading = False

    @classmethod
    def get_instance(cls) -> "SAM2Service":
        if cls._instance is None:
            cls._instance = SAM2Service()
        return cls._instance

    def _resolve_device(self) -> str:
        dev_cfg = settings.sam_device.lower()
        if dev_cfg == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif dev_cfg == "cpu":
            return "cpu"
        else: # auto
            return "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self) -> bool:
        if not self.enabled:
            logger.info("SAM 2.1 is disabled by configuration (SAM_ENABLED=false).")
            return False

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

            logger.info("SAM 2.1 initialized")
            logger.info("Model: Hiera Tiny")
            logger.info(f"Device: {self.device.upper()}")
            logger.info(f"SAM model loaded successfully in {load_ms:.1f}ms.")
            return True
        except Exception as e:
            self._is_loading = False
            self.ready = False
            logger.error(f"Failed to initialize SAM 2.1 model: {e}", exc_info=True)
            return False

    def get_info(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "model": "sam2.1_hiera_tiny",
            "model_id": self.model_id,
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "ready": self.ready
        }

    def segment_image(
        self,
        image: Image.Image,
        bbox: NormalizedBBox,
        points: Optional[List[Tuple[float, float, int]]] = None
    ) -> Dict[str, Any]:
        """
        Runs SAM 2.1 segmentation for a single image and normalized bounding box.
        Returns base64 PNG mask, IoU confidence score, dimensions, and timing.
        """
        if not self.ready:
            loaded = self.load_model()
            if not loaded:
                raise RuntimeError("SAM 2.1 service is not ready and could not be loaded.")

        t0 = time.time()
        w, h = image.size

        # Convert normalized bbox [x_min, y_min, x_max, y_max] to pixel coordinates [x1, y1, x2, y2]
        x1 = max(0.0, min(float(w), bbox.x_min * w))
        y1 = max(0.0, min(float(h), bbox.y_min * h))
        x2 = max(0.0, min(float(w), bbox.x_max * w))
        y2 = max(0.0, min(float(h), bbox.y_max * h))

        # Ensure box has minimum size
        if x2 - x1 < 2:
            x2 = min(float(w), x1 + 2)
        if y2 - y1 < 2:
            y2 = min(float(h), y1 + 2)

        # Prepare inputs: 3 levels nesting [image level, box level, box coords]
        inputs = self.processor(
            images=image,
            input_boxes=[[[x1, y1, x2, y2]]],
            return_tensors="pt"
        )

        if self.device != "cpu":
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process masks back to original image size
        # inputs['original_sizes'] is list of [H, W]
        post_masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            binarize=True
        )

        # outputs.iou_scores: shape [batch_size, num_boxes, 3]
        iou_scores = outputs.iou_scores[0, 0].cpu().numpy()
        best_candidate_idx = int(np.argmax(iou_scores))
        best_iou = float(iou_scores[best_candidate_idx])

        # Extract selected mask [1, 3, H, W] -> [H, W]
        mask_tensor = post_masks[0][0, best_candidate_idx]
        mask_np = mask_tensor.numpy().astype(np.uint8) * 255

        fg_count = int(np.count_nonzero(mask_np))

        # Encode as 1-bit or 8-bit PNG
        mask_pil = Image.fromarray(mask_np, mode="L")
        buf = io.BytesIO()
        mask_pil.save(buf, format="PNG", optimize=True)
        b64_str = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

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
    return SAM2Service.get_instance()
