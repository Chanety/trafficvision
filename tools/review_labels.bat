@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set PY=trafficlab-3d\env\Scripts\python.exe
if not exist "%PY%" (
  echo 找不到 Python 环境：%PY%
  pause
  exit /b 1
)
"%PY%" tools\start_labelimg.py
if errorlevel 1 pause
