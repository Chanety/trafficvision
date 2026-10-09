@echo off
cd /d "%~dp0.."
echo.
echo  ================================================
echo   自留复核  videoset_02_label3（10 类，已清洗）
echo   你的部分是优先级最前的 100 帧，名单见
echo   dataset_batches\自留帧名单.txt
echo  ================================================
echo.
echo  空白拖动=画框   点框=选中   数字键=改类别(1-9,0=第10类)
echo  Delete=删框     A/左=上一张   D/右=下一张(自动保存)
echo  R=清骑车人      Ctrl+Z=撤销   Ctrl+S=保存当前
echo.
"trafficlab-3d\env\Scripts\python.exe" tools\annotate_zh.py datasets\videoset_02_label3
echo.
echo  退出码 %ERRORLEVEL%（0 为正常）
pause
