@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem ---- KGED Stage 5 one-click engineering prediction runner (Windows) ----
rem Set KGED_NO_PAUSE=1 to suppress the final pause (CI / scripted use).
set "EXIT_CODE=0"
pushd "%~dp0" || (echo [ERROR] Cannot enter the repository directory.& set "EXIT_CODE=1"& goto :finish)
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (py -3 -c "import sys" >nul 2>&1 && set "PY=py -3")
if not defined PY (python -c "import sys" >nul 2>&1 && set "PY=python")
if not defined PY (
    echo [ERROR] Python 3.9 or newer was not found.
    echo         Install Python from https://www.python.org/downloads/ and enable "Add python.exe to PATH".
    set "EXIT_CODE=9009"
    goto :finish
)

%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)"
if errorlevel 1 (
    echo [ERROR] Python 3.9 or newer is required.
    set "EXIT_CODE=1"
    goto :finish
)

%PY% -c "import numpy, scipy, matplotlib" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing required packages from requirements.txt ...
    %PY% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed. Please run: python -m pip install -r requirements.txt
        set "EXIT_CODE=1"
        goto :finish
    )
)

%PY% "%~dp0run_engineering_predictions.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if "%EXIT_CODE%"=="0" (
    echo [OK] All engineering tables, figures, and run summary were written to engineering\results\
) else (
    echo [ERROR] Engineering prediction pipeline exited with code %EXIT_CODE%.
)

:finish
popd 2>nul
rem Pause only when started by double-click (cmd /c ...); delayed expansion keeps & ) ^ in paths safe.
setlocal EnableDelayedExpansion
set "_CL=!CMDCMDLINE!"
if not defined KGED_NO_PAUSE if /i not "!_CL:/c =!"=="!_CL!" pause
endlocal & exit /b %EXIT_CODE%
