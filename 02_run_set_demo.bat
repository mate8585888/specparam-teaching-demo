@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Local environment was not found.
    echo Please double-click 01_install.bat first.
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" "app\fooof_set_teaching.py"

echo.
pause
