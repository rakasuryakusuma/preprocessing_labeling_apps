@echo off
setlocal EnableDelayedExpansion
title OCT Labeling Server - Akses dari Tablet via Wi-Fi

cd /d "%~dp0"

echo ============================================================================
echo   OCT Preprocessing ^& Labeling Web Server (Akses via Tablet)
echo ============================================================================
echo.

:: 1. Cek apakah ada virtual environment lokal (.venv)
if exist "%~dp0.venv\Scripts\python.exe" (
    echo [INFO] Menggunakan local virtual environment ^(.venv^)...
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
    goto :CHECK_DEPS
)

:: 2. Cek apakah Miniconda3 tersedia di Drive D atau profil pengguna
if exist "D:\miniconda3\Scripts\activate.bat" (
    echo [INFO] Mengaktifkan Miniconda3 dari D:\miniconda3 ...
    call "D:\miniconda3\Scripts\activate.bat"
    set "PYTHON_EXE=python"
    goto :CHECK_DEPS
)

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
echo [INFO] Memeriksa dependensi ^(opencv-python, numpy^)...
"%PYTHON_EXE%" -c "import cv2, numpy" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [INFO] Menginstal dependensi dari requirements.txt ...
    "%PYTHON_EXE%" -m pip install --no-cache-dir -r "%~dp0requirements.txt"
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Gagal menginstal dependensi.
        pause
        exit /b 1
    )
)

echo [INFO] Menjalankan server tablet_server.py ...
echo.
"%PYTHON_EXE%" "%~dp0tablet_server.py"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Server berhenti dengan kode error %ERRORLEVEL%.
    pause
)

endlocal
