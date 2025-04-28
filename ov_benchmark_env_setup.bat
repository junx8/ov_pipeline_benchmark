@echo off
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Python already installed.
) else (
    echo Please install Python.
    exit /b 10
)

@echo off
python --version 2>&1 | findstr /C:"Python 3.12" >nul
if %errorlevel% equ 0 (
    echo The installed python is 3.12
) else (
    echo Please install Python==3.12
    @REM python 3.13 is not support fine for onnx 
    exit /b 10
)

echo.
echo Start creating virtual environment...
python -m venv ov_env
call ov_env\Scripts\activate.bat

set http_proxy=http://child-prc.intel.com:913
set https_proxy=http://child-prc.intel.com:913

echo.
echo Installing required packages...
@REM python -m pip install --upgrade pip
pip install anomalib
anomalib install --option full -v

echo.
echo Prepare the MVTecAD datasets...
echo ### Or you can create the directory [datasets/MVTecAD] and put the mvtec data into it manually.
echo.

echo If this is the first time you are running it, please download the MVTecAD datasets for AI Benchmark.
choice /C YN /M "Do you want download the MVTecAD datasets?"

if errorlevel 2 (
    echo.
    echo.
    echo Select not download the MVTecAD datasets for AI Benchmark...
    echo Please make sure put the MVTecAD datasets into the AI Benchmark directory. 
) else if errorlevel 1 (
    echo.
    echo.
    echo Starting download the MVTecAD datasets for AI Benchmark...
    python prepare_dataset.py
)


echo.
echo.
echo Done!
echo Exit of the virtual environment...
call ov_env\Scripts\deactivate.bat