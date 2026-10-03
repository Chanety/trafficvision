@echo off
cd /d "%~dp0.."
"trafficlab-3d\env\Scripts\python.exe" tools\start_labelimg.py
echo.
echo program exited, errorlevel = %errorlevel%
pause
