@echo off
chcp 65001 >nul
setlocal
call "%~dp0_batch_config.bat"
cd /d "%~dp0..\.."

set "BATCH=batch_02"
set "DATA=datasets\%BATCH_NAME%\%BATCH%"
if not exist "%DATA%\images" (
  echo [缺少数据] 还没解压 %BATCH%。把 batch_02.zip 放到 tools\dataset_batches\ 后双击 解压批次.bat
  pause
  exit /b 1
)

set "PY=trafficlab-3d\env\Scripts\python.exe"
if not exist "%PY%" (
  where python >nul 2>nul
  if errorlevel 1 (
    echo [缺少标注工具运行环境] 本机既没有项目的 Python 环境，也没装 Python。
    echo    · 装上 Python 与依赖：pip install PyQt5 opencv-python numpy
    echo    · 或者用任何 YOLO 格式标注工具直接打开：
    echo      %CD%\%DATA%\images  与  %CD%\%DATA%\labels
    echo      （规则见 %CD%\%DATA%\标注说明.txt，classes.txt 是类别表）
    pause
    exit /b 1
  )
  set "PY=python"
)

echo 正在打开 %DATA% 的复核标注...
echo 滚轮缩放 / 中键或空格拖动 / 拖角改框 / R 清理重叠行人框 / Ctrl+Z 撤销
echo 改完一帧按 D 或右方向键进入下一帧，进度实时写入 labels\
echo.
"%PY%" tools\annotate_zh.py "%DATA%"
if errorlevel 1 (
  echo.
  echo [提示] 标注工具没能启动。多半是缺少 PyQt5 / opencv-python：
  echo        "%PY%" -m pip install PyQt5 opencv-python numpy
  echo        或者换用任何 YOLO 格式标注工具打开上面的 images / labels 目录。
)
echo.
echo program exited, errorlevel = %errorlevel%
pause
