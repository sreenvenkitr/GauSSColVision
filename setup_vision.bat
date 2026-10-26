@echo off
echo ===================================================
echo Setting up GauSSColVision Python Environment
echo ===================================================

python --version
if errorlevel 1 (
    echo Error: Python is not found on PATH. Please install Python 3.10+ and add to PATH.
    pause
    exit /b 1
)

echo Installing dependencies from requirements.txt...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Error during package installation.
    pause
    exit /b 1
)

echo.
echo GauSSColVision setup complete.
echo You can now run the service using run_server.bat
pause
