@echo off
chcp 65001 >nul
setlocal
call "%~dp0_batch_config.bat"
cd /d "%~dp0..\.."

set "N="
set /p N=请输入你复核的批次号（1 / 2 / 3）后回车：
set "N=%N: =%"
if not defined N (
  echo [没有输入] 请重新双击本程序，输入 1 / 2 / 3 中的一个。
  pause
  exit /b 1
)
if not "%N%"=="1" if not "%N%"=="2" if not "%N%"=="3" (
  echo [批次号不对] 只能是 1 / 2 / 3。你输入的是：%N%
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pack_results.ps1" ^
  -Batch "batch_0%N%" -DataName "%BATCH_NAME%"
echo.
pause
