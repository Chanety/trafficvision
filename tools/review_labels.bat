@echo off
cd /d "%~dp0.."
echo ============================================================
echo  注意：这是旧版英文工具 labelImg，不支持中文界面，
echo        也没有"先载具后行人"规则和拖动平移。
echo        日常复核请使用 tools\annotate_zh.bat（中文版）
echo ============================================================
echo.
"trafficlab-3d\env\Scripts\python.exe" tools\start_labelimg.py
echo.
echo program exited, errorlevel = %errorlevel%
pause
