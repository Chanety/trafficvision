@echo off
chcp 65001 >nul
cd /d "%~dp0.."
"trafficlab-3d\env\Scripts\python.exe" tools\annotate_zh.py
echo.
echo program exited, errorlevel = %errorlevel%
pause
