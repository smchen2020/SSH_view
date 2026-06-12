@echo off
title SSH and Ocean Currents Visualizer Startup
color 0A
echo ===================================================
echo   Starting SSH and Ocean Currents Visualizer...
echo   The browser window will open automatically.
echo ===================================================
echo.

:: Execute uvicorn using the dedicated conda environment python binary
"C:\Users\2011018\AppData\Local\anaconda3\envs\ssh\python.exe" -m uvicorn app:app --port 8000 --reload
pause
