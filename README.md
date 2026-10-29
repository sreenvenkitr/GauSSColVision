# GauSSColVision

Multimodal Visual Reasoning & Inspection Service for GauSSCol 3D Gaussian Splatting scenes using **Google Gemini** and **SAM 2.1**.

## Architecture

```text
GauSSCol Browser Application
    |
    | HTTP POST /api/v1/vision/query
    v
GauSSColVision Service (Port 8010)
    |
    +---> VisualReasoningService
            |
            +---> VisionLanguageModel (Google Gemini Cloud API / Mock Fallback)
            |
            +---> SAM2Service (SAM 2.1 Hiera Tiny 2D Neural Segmentation)
```

## Prerequisites

1. **Python**: 3.10 to 3.14 (64-bit).
2. **PyTorch**: 2.0+ (CUDA GPU supported, CPU fallback available for SAM 2.1).
3. **Gemini API Key**: Set in `.env` as `GEMINI_API_KEY=...`
4. **GauSSCol Web UI**: Run locally from [GauSSCol-Demo](https://github.com/sreenvenkitr/GauSSCol-Demo) via `run_demo.bat` (`http://localhost:8000`).

## Setup Instructions

### 1. Install Dependencies
Run in PowerShell / Command Prompt:
```cmd
setup_vision.bat
```
Or manually:
```cmd
python -m pip install -r requirements.txt
```

### 2. Start the Vision Server
```cmd
run_server.bat
```
The server will start on: `http://127.0.0.1:8010`

### 3. Verify Health
Open in browser or run:
```powershell
Invoke-RestMethod http://127.0.0.1:8010/health
```
Example response:
```json
{
  "status": "ok",
  "model": "Google Gemini (gemini-3.6-flash)",
  "device": "cloud_api",
  "cuda_available": false,
  "dtype": "fp16_cloud",
  "ready": true,
  "sam_enabled": true,
  "sam_model": "facebook/sam2.1-hiera-tiny",
  "sam_ready": true
}
```

### 4. Run Test Suite
```powershell
python -m pytest tests/
```

## API Endpoints

- `GET /health`: Model status, device, precision, readiness.
- `POST /api/v1/vision/query`: Primary multimodal query endpoint accepting multipart form-data with metadata JSON and rendered splat views.
