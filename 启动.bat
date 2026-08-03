@echo off
rem 双击 bat 时 cwd 默认是 C:\Windows\System32，不切过来沙箱会锁错盘
cd /d %~dp0
chcp 65001 >nul

rem 优先用项目自带的 venv，避免污染系统 Python
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY=%~dp0.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

if not exist "%~dp0config.json" (
    echo 没有 config.json，正在从 config.example.json 复制一份……
    copy /y "%~dp0config.example.json" "%~dp0config.json" >nul
    echo 请打开 config.json 填入千问 api_key 和 work_dir，然后重新运行。
    pause
    exit /b 1
)

"%PY%" main.py
if errorlevel 1 (
    echo.
    echo 启动失败。若提示缺少依赖，先运行：
    echo     python -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
)
pause
