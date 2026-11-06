@echo off
setlocal EnableDelayedExpansion
title Server Tablet Redmi Pad 2 - OCT Labeler

cd /d "%~dp0"

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat"
    set "PYTHON_EXE=python"
) else if exist "%LOCALAPPDATA%\miniconda3\Scripts\activate.bat" (
    call "%LOCALAPPDATA%\miniconda3\Scripts\activate.bat"
    set "PYTHON_EXE=python"
) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
    set "PYTHON_EXE=python"
) else (
    set "PYTHON_EXE=python"
)

echo ============================================================================
echo   Mode Tablet ^(Redmi Pad 2 + Stylus Pen^)
echo ============================================================================
echo.
echo Masukkan path Folder Sumber OCT ^(atau tekan ENTER untuk memakai folder input_oct^):
set /p IN_DIR="Folder Sumber: "
echo.
echo Masukkan path Folder Output ^(atau tekan ENTER untuk memakai folder output_oct^):
set /p OUT_DIR="Folder Output: "
echo.

set "ARGS="
if not "%IN_DIR%"=="" set "ARGS=!ARGS! --input "%IN_DIR%""
if not "%OUT_DIR%"=="" set "ARGS=!ARGS! --output "%OUT_DIR%""

"%PYTHON_EXE%" "%~dp0tablet_server.py" !ARGS!

pause
endlocal
