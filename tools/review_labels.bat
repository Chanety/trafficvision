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
echo.
echo 程序已退出（退出码 %errorlevel%）。如非 0，请把上面的信息截图发我。
pause
