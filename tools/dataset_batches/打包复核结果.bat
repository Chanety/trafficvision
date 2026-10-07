@echo off
chcp 65001 >nul
setlocal
call "%~dp0_batch_config.bat"
cd /d "%~dp0..\.."

set "N="
set /p N=请输入你复核的批次号（1 / 2 / 3）后回车：
if not defined N ( echo 没有输入批次号。 & pause & exit /b 1 )

set "BATCH=batch_0%N%"
set "SRC=datasets\%BATCH_NAME%\%BATCH%"
if not exist "%SRC%\labels" (
  echo [找不到] %SRC%\labels 不存在，批次号对吗？或者还没下载数据？
  pause
  exit /b 1
)

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set "TODAY=%%i"
set "OUT=复核结果_%BATCH%_%TODAY%.zip"

echo 正在打包 %SRC%\labels ...
powershell -NoProfile -Command ^
  "Compress-Archive -Path '%SRC%\labels','%SRC%\review_order.txt','%SRC%\标注说明.txt' -DestinationPath '%OUT%' -Force"

if not exist "%OUT%" (
  echo [失败] 打包没成功，请把上面的报错截图发给负责人。
  pause
  exit /b 1
)

echo.
echo [完成] 已生成：%CD%\%OUT%
echo        把这个压缩包发回给负责人即可（里面只有你改过的标注，不含图片）。
echo.
pause
