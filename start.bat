@echo off
title Local Dream
pushd "%~dp0"

echo Stopping existing processes...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8081 ^| findstr LISTENING') do (
    echo Killing PID %%a on port 8081
    taskkill /f /pid %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5173 ^| findstr LISTENING') do (
    echo Killing PID %%a on port 5173
    taskkill /f /pid %%a >nul 2>&1
)

timeout /t 3 /nobreak >nul

echo Starting backend on port 8081...
pushd web\backend
start /b python run.py
popd

timeout /t 5 /nobreak >nul

echo Starting frontend on port 5173...
pushd web
start /b npm run dev
popd

echo.
echo ==========================================
echo   Frontend: http://localhost:5173
echo   Backend:  http://127.0.0.1:8081 (LAN: http://YOUR_IP:8081)
echo ==========================================
pause
popd
