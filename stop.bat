@echo off
title APEX - Stop Services
echo.
echo [INFO] Stopping APEX services...
taskkill /FI "WINDOWTITLE eq APEX Backend*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq APEX Frontend*" /F >nul 2>&1
taskkill /F /IM "uvicorn.exe" >nul 2>&1
echo [INFO] Services stopped.
timeout /t 2 /nobreak >nul
