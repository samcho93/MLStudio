@echo off
chcp 65001 >nul
title ML Node Studio - Launcher

echo ============================================
echo   ML Node Studio - Starting...
echo ============================================
echo.

set "PYTHON=%USERPROFILE%\AppData\Local\Programs\Python\Python39\python.exe"
set "BACKEND=D:\Work\WebSharp\backend"
set "FRONTEND=D:\Work\WebSharp\frontend-web"

echo [1/3] Starting Backend (FastAPI :8000)...
start "ML-Backend" cmd /k "cd /d %BACKEND% && "%PYTHON%" -X utf8 main.py"

timeout /t 2 /nobreak >nul

echo [2/3] Starting Frontend (Vite :5173)...
start "ML-Frontend" cmd /k "cd /d %FRONTEND% && npm run dev -- --host"

timeout /t 3 /nobreak >nul

echo [3/3] Opening Browser...
start http://localhost:5173

echo.
echo ============================================
echo   All services started!
echo ============================================
echo.
echo   Local:    http://localhost:5173
echo   Backend:  http://localhost:8000
echo.
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr "IPv4"') do (
    for /f "tokens=1" %%b in ("%%a") do (
        echo   Network:  http://%%b:5173
    )
)
echo.
echo   External: run start_tunnel.bat (ngrok)
echo.
echo ============================================
echo   Press any key to stop all servers...
echo ============================================
pause >nul

echo.
echo Stopping servers...
taskkill /fi "WINDOWTITLE eq ML-Backend*" /f >nul 2>&1
taskkill /fi "WINDOWTITLE eq ML-Frontend*" /f >nul 2>&1
echo Done.
