@echo off
:: ============================================================
::  audio-ml-pipeline  --  Master Launcher
::  Kills all running pipeline instances then starts the
::  selected pipeline.
::
::  Usage:
::    start_pipeline.bat          -> ML Denoiser Pipeline (U-Net ONNX, default)
::    start_pipeline.bat b        -> DSP Filter Pipeline (scipy filters, fallback)
::    start_pipeline.bat list     -> List audio devices and exit
:: ============================================================

setlocal
set PROJECT=%~dp0
set PY=python

:: ── Parse argument ────────────────────────────────────────────────────────
set MODE=ml
if /i "%~1"=="b"    set MODE=dsp
if /i "%~1"=="list" set MODE=list

:: ── Kill ALL python processes running any pipeline script ─────────────────
echo [*] Stopping all running pipeline instances...
%PY% -c "
import subprocess

scripts = ['ml_denoiser_pipeline.py', 'dsp_filter_pipeline.py']
result = subprocess.run(
    ['wmic', 'process', 'where', 'name=\"python.exe\" or name=\"py.exe\"',
     'get', 'ProcessId,CommandLine', '/format:csv'],
    capture_output=True, text=True
)
killed = 0
for line in result.stdout.splitlines():
    for s in scripts:
        if s in line:
            try:
                pid = int(line.strip().split(',')[-1])
                subprocess.run(['taskkill', '/F', '/PID', str(pid)], capture_output=True)
                print(f'  Killed PID {pid} ({s})')
                killed += 1
            except Exception:
                pass
if killed == 0:
    print('  No running pipeline instances found.')
"
timeout /t 2 /nobreak >nul

:: ── Dispatch ─────────────────────────────────────────────────────────────
if "%MODE%"=="list" (
    echo.
    echo [Audio Devices]
    %PY% "%PROJECT%list_devices.py"
    pause
    goto :eof
)

if "%MODE%"=="dsp" (
    echo [*] Launching DSP Filter Pipeline...
    start "dsp_filter_pipeline.py" cmd /k "%PY% "%PROJECT%dsp_filter_pipeline.py""
    goto :done
)

echo [*] Launching ML Denoiser Pipeline...
start "ml_denoiser_pipeline.py" cmd /k "%PY% "%PROJECT%ml_denoiser_pipeline.py""

:done
echo [*] Done.
endlocal
