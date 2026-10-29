import logging
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .api.vision import router as vision_router
from .api.sam import router as sam_router
from .models.gemini import get_vlm_model
from .services.sam2_service import get_sam2_service

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("gausscol_vision")

app = FastAPI(
    title="GauSSColVision API",
    description="Multimodal Visual Reasoning & Inspection Service for GauSSCol 3D Gaussian Splatting",
    version="1.0.0"
)

# Enable CORS for browser frontends (including Private Network Access for HTTPS sites)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_private_network=True,
)

# Register endpoints
app.include_router(vision_router)
app.include_router(sam_router)

@app.on_event("startup")
async def startup_event():
    logger.info("==================================================")
    logger.info("Starting GauSSColVision Service")
    logger.info(f"Host: {settings.host}:{settings.port}")
    if settings.vision_provider in ("openrouter", "openai", "openai_compatible"):
        logger.info(f"VLM Provider: OpenAI-Compatible / OpenRouter ({settings.openrouter_model} at {settings.openrouter_base_url})")
    else:
        logger.info(f"VLM Provider: Gemini Cloud API ({settings.gemini_model})")
    logger.info(f"Mock Mode: {settings.use_mock_model}")
    logger.info(f"SAM 2.1 Enabled: {settings.sam_enabled}")
    logger.info("==================================================")
    # Warm up model initialization
    try:
        model = get_vlm_model()
        if hasattr(model, 'load') and not getattr(model, 'ready', False):
            model.load()
        info = model.get_info()
        logger.info(f"VLM initialized: {info['model']} (Device: {info['device']}, Ready: {info['ready']})")
    except Exception as e:
        logger.error(f"Error during startup model initialization: {e}")

    # Warm up SAM 2.1 if enabled
    if settings.sam_enabled:
        try:
            sam_service = get_sam2_service()
            sam_service.load_model()
        except Exception as e:
            logger.warning(f"SAM 2.1 startup warmup warning: {e}")

