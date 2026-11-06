@echo off
setlocal EnableDelayedExpansion
title Verifikasi Dataset Masker OCT (Milestone 4)

cd /d "%~dp0"

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat"
    set "PYTHON_EXE=python"
) else (
    set "PYTHON_EXE=python"
)

if "%~1"=="" (
    set /p OUT_DIR="Masukkan path folder output (contoh: D:\data\output): "
) else (
    set "OUT_DIR=%~1"
)

"%PYTHON_EXE%" "%~dp0verify_dataset.py" "%OUT_DIR%" --preview
pause
endlocal
