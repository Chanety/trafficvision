@echo off
chcp 65001 >nul
setlocal
rem 切到项目根目录（本文件在 tools\ 下）
cd /d "%~dp0.."
set "PY=trafficlab-3d\env\Scripts\python.exe"

if not exist "%PY%" (
  echo [错误] 找不到项目虚拟环境：
  echo        %CD%\%PY%
  echo        请确认项目目录完整，或先按 README 重建环境。
  pause
  exit /b 1
)

echo 启动图形界面...
echo   解释器：%PY%
echo   工作目录：%CD%
echo.
"%PY%" -m trafficvision_gui.main

if errorlevel 1 (
  echo.
  echo [提示] 界面退出时返回了非零码。常见原因：
  echo        · 依赖未安装：pip install -r requirements-gui-api.txt
  echo        · 没有图形显示环境（远程/无桌面会话）
  pause
)
endlocal
