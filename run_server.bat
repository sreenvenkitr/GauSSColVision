@echo off
echo ===================================================
echo Starting GauSSColVision FastAPI Server on port 8010
echo ===================================================

python -m uvicorn app.main:app --host 127.0.0.1 --port 8010 --reload
pause
