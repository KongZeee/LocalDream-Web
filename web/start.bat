@echo off
echo ============================================
echo   Local Dream - Starter
echo ============================================
echo.
echo [1] Starting Backend API Server...
echo [2] Starting Frontend Dev Server...
echo.

start "Local Dream Backend" cmd /k "cd /d %~dp0backend && python run.py"
timeout /t 2 >nul
start "Local Dream Frontend" cmd /k "cd /d %~dp0 && npm run dev"

echo.
echo Both servers are starting...
echo Frontend: http://localhost:5173
echo Backend:  http://127.0.0.1:8081
echo.
echo Press any key to close this window...
pause >nul