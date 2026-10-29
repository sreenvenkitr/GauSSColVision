@echo off
echo ===================================================
echo Starting GauSSColVision FastAPI Server on port 8010
echo ===================================================

if not exist .env (
    echo [WARNING] No .env file found!
    echo Creating .env from .env.example...
    copy .env.example .env >nul
    echo.
    echo IMPORTANT: Please edit .env and insert your GEMINI_API_KEY!
    echo Without a valid GEMINI_API_KEY, the server runs in Mock Mode
    echo (offline fallback) which uses dummy bounding boxes.
    echo ===================================================
    echo.
)

python -m uvicorn app.main:app --host 127.0.0.1 --port 8010 --reload
pause
