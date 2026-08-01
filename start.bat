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
    call bun install
    cd ..
    set FRONTEND_READY=1
)

echo.
echo [INFO] Starting Pure Rust APEX Server on http://localhost:8080 ...
start "APEX Rust Server" cmd /k "cd backend\rust_core && .\target\release\apex_server.exe"

echo [INFO] Waiting for server to start...
timeout /t 2 /nobreak >nul

echo [INFO] Starting Bun React Frontend on http://localhost:5173 ...
start "APEX Frontend" cmd /k "cd frontend && bun run dev"

echo.
echo  ============================================
echo   App is running (Pure Rust & Bun)!
echo.
echo   Rust Backend API :  http://localhost:8080
echo   Bun Frontend UI  :  http://localhost:5173
echo   Health Status    :  http://localhost:8080/health
echo  ============================================
echo.
echo  Press any key to open the app in your browser...
pause >nul

start http://localhost:5173
