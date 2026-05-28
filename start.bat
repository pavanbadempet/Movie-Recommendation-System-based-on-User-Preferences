@echo off
title APEX Recommendation System
color 0A

echo.
echo  ============================================
echo   APEX Recommendation System - Launcher
echo  ============================================
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [WARN] No venv found. Using system Python.
)

REM Check if node_modules exists for React frontend
if exist "frontend\node_modules" (
    echo [INFO] React frontend found.
    set FRONTEND_READY=1
) else (
    echo [WARN] React node_modules not found.
    echo [INFO] Installing frontend dependencies...
    cd frontend
    call npm install
    cd ..
    set FRONTEND_READY=1
)

echo.
echo [INFO] Starting FastAPI Backend on http://localhost:8000 ...
start "APEX Backend" cmd /k "python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

echo [INFO] Waiting for backend to start...
timeout /t 4 /nobreak >nul

echo [INFO] Starting React Frontend on http://localhost:5173 ...
start "APEX Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo  ============================================
echo   App is running!
echo.
echo   Backend API  :  http://localhost:8000
echo   Frontend UI  :  http://localhost:5173
echo   API Docs     :  http://localhost:8000/docs
echo  ============================================
echo.
echo  Press any key to open the app in your browser...
pause >nul

start http://localhost:5173
