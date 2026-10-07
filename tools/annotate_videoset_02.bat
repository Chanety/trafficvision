@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo 正在打开 videoset_02 的复核标注（332 帧，按"三轮车→自行车→标志牌→路桩→公交/卡车"优先排序）
echo 滚轮缩放 / 中键或空格拖动 / 拖角改框 / R 清理重叠行人框 / Ctrl+Z 撤销
echo 复核完一帧按 D 或右方向键进入下一帧，进度会实时写入 datasets\videoset_02_label\labels
echo.
"trafficlab-3d\env\Scripts\python.exe" tools\annotate_zh.py datasets\videoset_02_label
echo.
echo program exited, errorlevel = %errorlevel%
pause
