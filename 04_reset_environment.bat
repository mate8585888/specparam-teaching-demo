@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo This will delete only the .venv folder inside this teaching package.
echo Your EEG data and output results will not be deleted.
echo.
set /p CONFIRM=Type YES to continue: 
if /I not "%CONFIRM%"=="YES" (
    echo Cancelled.
    pause
    exit /b 0
)

if exist ".venv" (
    rmdir /s /q ".venv"
    echo Local environment deleted.
) else (
    echo No local environment was found.
)

pause
