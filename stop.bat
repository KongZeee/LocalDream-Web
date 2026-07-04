@echo off
title Local Dream - Stop
pushd "%~dp0"

echo Stopping Local Dream processes...

set STOPPED=0
for %%P in (8081 5173) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%%P ^| findstr LISTENING') do (
        echo Killing PID %%a on port %%P
        taskkill /f /pid %%a >nul 2>&1
        set STOPPED=1
    )
)

if %STOPPED%==0 (
    echo No running processes found on ports 8081 / 5173.
)

echo.
echo Done.
pause
popd
