@echo off
rem Double-click to run RoRo with the parameters set in run_roro.py.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [x] .venv not found. Open a terminal in this folder and run: uv sync
  pause
  exit /b 1
)
".venv\Scripts\python.exe" run_roro.py
echo.
pause
