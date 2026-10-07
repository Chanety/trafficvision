@echo off
chcp 65001 >nul
setlocal
call "%~dp0_batch_config.bat"
cd /d "%~dp0..\.."

set "BATCH=batch_02"
set "DATA=datasets\%BATCH_NAME%\%BATCH%"
if not exist "%DATA%\images" (
  echo [缺少数据] 还没下载 %BATCH%，请先双击 下载批次02.bat
  pause
  exit /b 1
)

set "PY=trafficlab-3d\env\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo 正在打开 %DATA% 的复核标注...
echo 滚轮缩放 / 中键或空格拖动 / 拖角改框 / R 清理重叠行人框 / Ctrl+Z 撤销
echo 改完一帧按 D 或右方向键进入下一帧，进度实时写入 labels\
echo.
"%PY%" tools\annotate_zh.py "%DATA%"
echo.
echo program exited, errorlevel = %errorlevel%
pause
