@echo off
cd /d "%~dp0.."
echo.
echo  ============================================================
echo   独立人工真值 · 帧对标注
echo.
echo   规则：每组（a/b 两张图）里，同一类别最多只标一个目标
echo         优先标明显在动的车/骑车人；别标停着的车与静态结构
echo.
echo   操作：空白拖动=画框   点框=选中   拖四角=改大小
echo         数字键 1-9 / 0=改类别   Delete=删框   Ctrl+Z=撤销
echo         A / 左=上一张   D / 右=下一张（自动保存）
echo.
echo   共 18 组 36 张，先做前 5 组也可以（10 分钟出第一份结果）
echo  ============================================================
echo.
"trafficlab-3d\env\Scripts\python.exe" tools\annotate_zh.py datasets\truth_pairs
echo.
echo  退出码 %ERRORLEVEL%（0 为正常）
pause
