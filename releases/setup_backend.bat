@echo off
rem Robodimm PRO local backend startup script for Windows
title Robodimm PRO Backend

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

echo === Setting up Robodimm PRO Local Backend ===

rem 1. Detect conda or mamba
where mamba >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set CONDA_EXE=mamba
    goto :found_conda
)

where conda >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set CONDA_EXE=conda
    goto :found_conda
)

rem Check default paths if not in PATH
if exist "%USERPROFILE%\miniforge3\Scripts\conda.exe" (
    set CONDA_EXE="%USERPROFILE%\miniforge3\Scripts\conda.exe"
    goto :found_conda
)
if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" (
    set CONDA_EXE="%USERPROFILE%\anaconda3\Scripts\conda.exe"
    goto :found_conda
)

echo ❌ Error: Neither conda nor mamba detected.
echo Please install Miniforge or Anaconda first to run the backend.
echo Visit: https://github.com/conda-forge/miniforge#miniforge3
pause
exit /b 1

:found_conda
echo Using package manager: %CONDA_EXE%

rem 2. Create environment if it doesn't exist
set ENV_NAME=robodimm-pro-backend
call %CONDA_EXE% env list | findstr /R "\<%ENV_NAME%\>" >nul
if %ERRORLEVEL% neq 0 (
    echo Creating Conda environment '%ENV_NAME%' from environment.yml...
    call %CONDA_EXE% env create -f "%PROJECT_ROOT%\environment.yml"
) else (
    echo Conda environment '%ENV_NAME%' already exists.
)

rem 3. Launch backend
echo Starting FastAPI backend server on 127.0.0.1:8001...
call %CONDA_EXE% run -n %ENV_NAME% python "%PROJECT_ROOT%\backend\main.py"
pause
