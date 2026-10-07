@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo ============================================================
echo  行人真值专项复核（40 帧，行人标注最密集的帧）
echo  重点看：骑车的人被标成了 person（按规则应只标载具）
echo  操作：滚轮缩放 / 中键或空格拖动 / 拖角改框
echo        R = 清理与载具重叠的行人框（就是清骑车人用的）
echo        Ctrl+Z 撤销 / D 或右方向键下一帧
echo  改的是 datasets\person_check\labels（不影响主标注集）
echo ============================================================
echo.
"trafficlab-3d\env\Scripts\python.exe" tools\annotate_zh.py datasets\person_check
echo.
echo program exited, errorlevel = %errorlevel%
pause
