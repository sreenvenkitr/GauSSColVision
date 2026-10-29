@echo off
echo ===================================================
echo Starting GauSSColVision FastAPI Server on port 8010
echo ===================================================

if not exist .env (
    copy .env.example .env >nul
    echo Created .env from template. Please set your GEMINI_API_KEY in .env
)

python -m uvicorn app.main:app --host 127.0.0.1 --port 8010 --reload
pause
