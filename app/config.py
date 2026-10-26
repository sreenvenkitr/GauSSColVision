import os
import torch
from dataclasses import dataclass, field
from typing import List

def _load_dotenv_if_present():
    candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),
        os.path.expanduser("~/.env")
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k not in os.environ:
                                os.environ[k] = v
                        elif len(line) >= 20 and " " not in line and "GEMINI_API_KEY" not in os.environ:
                            # Bare API key without variable name prefix
                            os.environ["GEMINI_API_KEY"] = line.strip("'\"")
            except Exception:
                pass

_load_dotenv_if_present()

@dataclass
class Settings:
    # Gemini Cloud API support (Fast 1-2s inference)
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("VISION_GEMINI_MODEL", "gemini-3.6-flash")
    
    host: str = os.getenv("VISION_HOST", "127.0.0.1")
    port: int = int(os.getenv("VISION_PORT", "8010"))
    
    # Device and precision (used for SAM 2.1)
    device: str = os.getenv("VISION_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    dtype: str = os.getenv("VISION_DTYPE", "bfloat16")
    
    # Full neural VLM inference: False
    use_mock_model: bool = os.getenv("VISION_USE_MOCK_MODEL", "false").lower() in ("true", "1", "yes")
    
    # Inference parameters
    max_views_per_query: int = int(os.getenv("VISION_MAX_VIEWS", "16"))
    max_new_tokens: int = int(os.getenv("VISION_MAX_NEW_TOKENS", "96"))
    temperature: float = float(os.getenv("VISION_TEMPERATURE", "0.0"))
    
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    # SAM 2.1 Configuration
    sam_enabled: bool = os.getenv("SAM_ENABLED", "true").lower() in ("true", "1", "yes")
    sam_model_id: str = os.getenv("SAM_MODEL", "facebook/sam2.1-hiera-tiny")
    sam_device: str = os.getenv("SAM_DEVICE", "auto")
    sam_min_mask_confidence: float = float(os.getenv("SAM_MIN_MASK_CONFIDENCE", "0.5"))
    gemini_grounding_min_confidence: float = float(os.getenv("GEMINI_GROUNDING_MIN_CONFIDENCE", "0.5"))
    semantic_selection_min_view_support: int = int(os.getenv("SEMANTIC_SELECTION_MIN_VIEW_SUPPORT", "1"))
    show_sam_debug_overlay: bool = os.getenv("SHOW_SAM_DEBUG_OVERLAY", "false").lower() in ("true", "1", "yes")

settings = Settings()
