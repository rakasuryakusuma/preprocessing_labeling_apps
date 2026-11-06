@echo off
setlocal EnableDelayedExpansion
title OCT Preprocessing ^& Labeling App (Fase 1)

cd /d "%~dp0"

echo ============================================================================
echo   OCT Preprocessing ^& Semantic Segmentation Labeling App (Fase 1)
echo ============================================================================
echo.

:: 1. Cek apakah ada virtual environment lokal (.venv)
if exist "%~dp0.venv\Scripts\python.exe" (
    echo [INFO] Menggunakan local virtual environment ^(.venv^)...
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
    goto :CHECK_DEPS
)

:: 2. Cek apakah Miniconda3 tersedia di profil pengguna (%USERPROFILE%\miniconda3)
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    echo [INFO] Mengaktifkan Miniconda3 dari %USERPROFILE%\miniconda3 ...
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat"
    set "PYTHON_EXE=python"
    goto :CHECK_DEPS
)

:: 3. Cek lokasi umum Conda lainnya
if exist "%LOCALAPPDATA%\miniconda3\Scripts\activate.bat" (
    echo [INFO] Mengaktifkan Miniconda3 dari %LOCALAPPDATA%\miniconda3 ...
    call "%LOCALAPPDATA%\miniconda3\Scripts\activate.bat"
    set "PYTHON_EXE=python"
    goto :CHECK_DEPS
)

if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    echo [INFO] Mengaktifkan Anaconda3 dari %USERPROFILE%\anaconda3 ...
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
    set "PYTHON_EXE=python"
    goto :CHECK_DEPS
)

:: 4. Fallback ke python di PATH
where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_EXE=python"
    goto :CHECK_DEPS
)

echo [ERROR] Python atau Miniconda tidak ditemukan!
echo Pastikan Python 3.9+ atau Miniconda sudah terinstal di komputer Anda.
pause
exit /b 1

:CHECK_DEPS
echo [INFO] Memeriksa dependensi ^(PyQt6, opencv-python, numpy^)...
"%PYTHON_EXE%" -c "import PyQt6, cv2, numpy" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Menginstal dependensi dari requirements.txt ...
    "%PYTHON_EXE%" -m pip install -r "%~dp0requirements.txt"
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Gagal menginstal dependensi.
        pause
        exit /b 1
    )
)

echo [INFO] Menjalankan aplikasi label_app.py ...
echo.
"%PYTHON_EXE%" "%~dp0label_app.py"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Aplikasi berhenti dengan kode error %ERRORLEVEL%.
    pause
)

endlocal
