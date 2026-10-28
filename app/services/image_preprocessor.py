"""
Image Preprocessor Service for GauSSColVision.

Provides aspect-ratio preserving downscaling and image format normalization
prior to Vision-Language Model (VLM) ingestion.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from PIL import Image

logger = logging.getLogger("gausscol_vision.image_preprocessor")


@dataclass(frozen=True)
class ImageDimensions:
    """
    Value object representing 2D image dimensions with helper methods
    for aspect ratio and scale calculation.
    """
    width: int
    height: int

    @property
    def max_dimension(self) -> int:
        """Returns the larger of width or height."""
        return max(self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        """Calculates width / height ratio with zero-division safety."""
        return self.width / max(1.0, float(self.height))

    def is_larger_than(self, max_dim: int) -> bool:
        """Checks if either dimension exceeds the given threshold."""
        return self.max_dimension > max_dim

    def compute_scaled_dimensions(self, target_max_dim: int, min_dim: int = 28) -> Tuple[int, int]:
        """
        Calculates new (width, height) maintaining the exact aspect ratio
        such that max(new_width, new_height) == target_max_dim.

        Args:
            target_max_dim: Target pixel length for the longest edge.
            min_dim: Minimum allowed length for any edge (prevents degenerate 0px dimensions).

        Returns:
            Tuple of (scaled_width, scaled_height).
        """
        scale = target_max_dim / float(self.max_dimension)
        new_w = max(min_dim, int(round(self.width * scale)))
        new_h = max(min_dim, int(round(self.height * scale)))
        return new_w, new_h


class AspectPreservingResizer:
    """
    Single-responsibility component that scales an individual PIL Image
    to fit within a maximum dimension bounding box while strictly preserving aspect ratio.
    """

    def __init__(self, max_dimension: int = 384, min_dimension: int = 28):
        """
        Args:
            max_dimension: Upper bound for the longest image dimension.
            min_dimension: Lower bound for any dimension to satisfy vision transformer requirements.
        """
        self.max_dimension = max_dimension
        self.min_dimension = min_dimension

    def resize_if_needed(self, image: Image.Image) -> Image.Image:
        """
        Downscales the image if its largest dimension exceeds max_dimension.
        If the image is already within bounds, returns the original instance unmodified.

        Args:
            image: Input PIL Image.

        Returns:
            Resized or original PIL Image.
        """
        dims = ImageDimensions(width=image.width, height=image.height)
        if not dims.is_larger_than(self.max_dimension):
            return image

        target_w, target_h = dims.compute_scaled_dimensions(
            target_max_dim=self.max_dimension,
            min_dim=self.min_dimension
        )

        # Bilinear resampling offers an optimal balance between downsampling speed
        # and edge sharpness for neural feature extraction.
        return image.resize((target_w, target_h), Image.Resampling.BILINEAR)


class ImageFormatStandardizer:
    """
    Ensures image color mode conforms to standard RGB format.
    Converts RGBA, paletted, or grayscale images into canonical 3-channel RGB.
    """

    @staticmethod
    def standardize_mode(image: Image.Image) -> Image.Image:
        """
        Converts the image to RGB mode if not already in RGB.
        Handles transparency flattening against a neutral white background.
        """
        if image.mode == "RGB":
            return image

        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            # Flatten alpha channel onto white background to prevent black background artifacts
            bg = Image.new("RGB", image.size, (255, 255, 255))
            rgba_image = image.convert("RGBA")
            bg.paste(rgba_image, mask=rgba_image.split()[3])
            return bg

        return image.convert("RGB")


class ImagePreprocessor:
    """
    Orchestrates image resizing, format standardization, and preparation
    for multimodal Vision-Language Model inference.

    Why this class exists:
    Modern Vision-Language Models (such as Gemini 2.5/3.6 Flash) incur token generation costs
    and latency that scale directly with image pixel dimensions. High-resolution WebGPU canvas
    snapshots (e.g., 1920x1080) can cause significant inference latency (3-5s) without improving
    coarse object detection. Downscaling images to a canonical maximum dimension (default 384px)
    preserves structural contours, color, and scene geometry while keeping inference latency
    under 1.5 seconds.
    """

    def __init__(
        self,
        max_dimension: int = 384,
        resizer: Optional[AspectPreservingResizer] = None,
        standardizer: Optional[ImageFormatStandardizer] = None
    ):
        """
        Args:
            max_dimension: Maximum pixel size for the longest edge.
            resizer: Optional custom AspectPreservingResizer instance.
            standardizer: Optional custom ImageFormatStandardizer instance.
        """
        self.max_dimension = max_dimension
        self.resizer = resizer or AspectPreservingResizer(max_dimension=max_dimension)
        self.standardizer = standardizer or ImageFormatStandardizer()

    def preprocess_single(self, image: Image.Image) -> Image.Image:
        """
        Standardizes format and scales down a single image if needed.

        Args:
            image: Input PIL Image.

        Returns:
            Preprocessed PIL Image.
        """
        standardized = self.standardizer.standardize_mode(image)
        return self.resizer.resize_if_needed(standardized)

    def preprocess_images(self, images: Dict[str, Image.Image]) -> Dict[str, Image.Image]:
        """
        Preprocesses a dictionary of viewpoint images keyed by viewpoint identifier
        (e.g., 'current', 'yaw_plus_45', 'top_birds_eye').

        Args:
            images: Mapping of viewpoint ID to PIL Image.

        Returns:
            Mapping of viewpoint ID to preprocessed PIL Image.
        """
        processed_images: Dict[str, Image.Image] = {}
        for view_id, img in images.items():
            processed_images[view_id] = self.preprocess_single(img)

        return processed_images
