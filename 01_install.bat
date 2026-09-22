@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo SpecParam / FOOOF Teaching Package - Environment Installer
echo ============================================================
echo.
echo This installer creates an isolated .venv inside this folder.
echo It will not modify packages in your other Python projects.
echo.

set "PY_CMD="

py -3.11 --version >nul 2>&1
if not errorlevel 1 set "PY_CMD=py -3.11"

if not defined PY_CMD (
    py -3.12 --version >nul 2>&1
    if not errorlevel 1 set "PY_CMD=py -3.12"
)

if not defined PY_CMD (
    py -3.10 --version >nul 2>&1
    if not errorlevel 1 set "PY_CMD=py -3.10"
)

if not defined PY_CMD (
    py -3.9 --version >nul 2>&1
    if not errorlevel 1 set "PY_CMD=py -3.9"
)

if not defined PY_CMD (
    python -c "import sys; raise SystemExit(0 if (3,9) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1
    if not errorlevel 1 set "PY_CMD=python"
)

if not defined PY_CMD (
    echo [ERROR] Compatible Python was not found.
    echo Please install 64-bit Python 3.11 or 3.12, then run this file again.
    echo During Python installation, check "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

echo Python command: %PY_CMD%
%PY_CMD% --version
echo.

if exist ".venv\Scripts\python.exe" (
    echo Existing local environment found. It will be reused.
) else (
    echo [1/4] Creating local virtual environment...
    %PY_CMD% -m venv ".venv"
    if errorlevel 1 goto :failed
)

echo [2/4] Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :failed

echo [3/4] Installing required packages...
".venv\Scripts\python.exe" -m pip install -r "requirements.txt"
if errorlevel 1 goto :failed

echo [4/4] Verifying the environment...
".venv\Scripts\python.exe" "app\check_environment.py"
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo Installation completed successfully.
echo Next, double-click: 02_run_set_demo.bat
echo You may first run: 03_run_builtin_demo.bat
echo ============================================================
echo.
pause
exit /b 0

:failed
echo.
echo ============================================================
echo Installation failed.
echo Check the messages above. Common causes:
echo 1. Network cannot access PyPI.
echo 2. Python version is not 3.9-3.12.
echo 3. Antivirus or permissions blocked .venv creation.
echo ============================================================
echo.
pause
exit /b 1
