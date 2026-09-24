@echo off
echo    Starting MediaPipe Project Environment Setup for Windows...

:: 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not added to your Environment PATH.
    echo Please install Python 3.11 from python.org and try again.
    exit /b 1
)

:: 2. Create the virtual environment if it doesn't exist
if not exist "mp_env" (
    echo    Creating virtual environment 'mp_env'...
    python -m venv mp_env
) else (
    echo    Virtual environment 'mp_env' already exists.
)

:: 3. Activate environment and install locked dependencies
echo    Activating environment and installing requirements...
call mp_env\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo ====================================================
echo    Windows Setup complete! To start your environment, run:
echo    mp_env\Scripts\activate.bat
echo    python your_script.py
echo ====================================================
