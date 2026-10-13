@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0.."
set "PY=trafficlab-3d\env\Scripts\python.exe"

if not exist "%PY%" (
  echo [错误] 找不到项目虚拟环境：%CD%\%PY%
  pause
  exit /b 1
)

echo 在线分析接口启动中...
echo   健康检查： http://127.0.0.1:8600/api/health
echo   交互文档： http://127.0.0.1:8600/docs
echo   按 Ctrl+C 停止服务
echo.
"%PY%" -m trafficvision_api.server --host 127.0.0.1 --port 8600

if errorlevel 1 (
  echo.
  echo [提示] 启动失败。常见原因：
  echo        · 依赖未安装：pip install -r requirements-gui-api.txt
  echo        · 8600 端口被占用，可改用 --port 8601
  pause
)
endlocal
