from abc import ABC, abstractmethod
from typing import Dict, Any
from PIL import Image
from ..schemas.vision import VisionQueryRequest, VisionQueryResponse

class VisionLanguageModel(ABC):
    """
    Abstract Base Class for Vision Language Models in GauSSColVision.
    Enables modular multimodal VLM integration (Gemini, mock, or other VLMs)
    without changing the FastAPI routes or browser frontend.
    """
    
    @abstractmethod
    def analyze(self, request: VisionQueryRequest, images: Dict[str, Image.Image]) -> str:
        """
        Execute multimodal inference given the request and view_id -> PIL.Image dictionary.
        Returns the raw model text response (to be parsed by VisualReasoningService).
        """
        pass
        
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Return model metadata for health/telemetry.
        """
        pass
