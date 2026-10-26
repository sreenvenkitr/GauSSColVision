import logging
from typing import Dict
from PIL import Image

logger = logging.getLogger("gausscol_vision.image_preprocessor")

class ImagePreprocessor:
    """
    Handles image resizing, aspect-ratio preservation, and preparation for VLM inference.

    Why this class exists:
    Modern Vision-Language Models (such as Gemini 2.0 Flash) incur token generation costs and
    latency that scale with image pixel dimensions. High-resolution WebGPU canvas snapshots
    (e.g., 1920x1080) can cause significant inference latency without improving object detection.
    Downscaling images to a canonical maximum dimension (384px) preserves structural contours,
    color, and geometry while keeping inference latency under 1.5 seconds.
    """

    def __init__(self, max_dimension: int = 384):
        self.max_dimension = max_dimension

    def preprocess_images(self, images: Dict[str, Image.Image]) -> Dict[str, Image.Image]:
        """
        Scales down each viewpoint image so that its largest dimension does not exceed max_dimension,
        while strictly maintaining the original aspect ratio and RGB color profile.
        """
        processed_images = {}
        for view_id, img in images.items():
            max_side = max(img.width, img.height)
            if max_side > self.max_dimension:
                scale = self.max_dimension / float(max_side)
                new_w = max(28, int(img.width * scale))
                new_h = max(28, int(img.height * scale))
                processed_images[view_id] = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            else:
                processed_images[view_id] = img

        return processed_images
