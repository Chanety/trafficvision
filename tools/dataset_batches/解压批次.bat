@echo off
chcp 65001 >nul
setlocal
call "%~dp0_batch_config.bat"
set "ORIG=%CD%"
cd /d "%~dp0..\.."
set "ROOT=%CD%"

rem 把别人发来的 batch_NN.zip 解压到 datasets\%BATCH_NAME%\batch_NN\
rem 用法：双击本文件（自动找 zip），或把 zip 直接拖到本文件上。

rem 只认 batch_01 / batch_02 这两个外发批次包（batch_self_keep.zip 是项目方自留，不参与）
set "AUTO="
set "ZIP=%~1"
if defined ZIP if not exist "%ZIP%" if exist "%ORIG%\%ZIP%" set "ZIP=%ORIG%\%ZIP%"
if not defined ZIP (
  set "AUTO=1"
  for %%N in (01 02) do (
    for %%D in ("%~dp0." "%ROOT%" "%ROOT%\%BATCH_DIR%" "%USERPROFILE%\Desktop" "%ROOT%\下载") do (
      if not defined ZIP if exist "%%~fD\batch_%%N.zip" set "ZIP=%%~fD\batch_%%N.zip"
    )
  )
)
if not defined ZIP (
  echo [没找到批次包] 把 batch_01.zip / batch_02.zip 放到 tools\dataset_batches\，
  echo                再双击本文件；或者直接把 zip 文件拖到本文件上。
  pause
  exit /b 1
)

for %%F in ("%ZIP%") do set "BATCH=%%~nF"
set "DEST=%ROOT%\datasets\%BATCH_NAME%\%BATCH%"
if not exist "%ZIP%" (
  echo [找不到文件] %ZIP%
  pause
  exit /b 1
)
set "ANS="
if defined AUTO set /p ANS=自动找到：%ZIP%   回车或输入 y 继续，输入 n 取消：
if /i "%ANS%"=="n" (
  echo 已取消，未改动任何文件。
  pause
  exit /b 0
)

echo 批次包: %ZIP%
echo 解压到: %DEST%
echo.

set "OK=y"
if exist "%DEST%" set /p OK=[注意] 目标已存在，解压会覆盖同名文件（含你改过的 labels）。继续请输入 y：
if /i not "%OK%"=="y" (
  echo 已取消，未改动任何文件。
  pause
  exit /b 0
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -LiteralPath '%ZIP%' -DestinationPath '%DEST%' -Force"
if errorlevel 1 (
  echo [失败] 解压失败，检查 zip 是否完整、磁盘是否够用。
  pause
  exit /b 1
)

echo.
echo [完成] 解压好了。现在双击 打开标注_%BATCH%.bat 开始复核（没有对应 bat 就用任何 YOLO 标注工具打开）。
pause
