@echo off
rem ============================================================================
rem  setup.cmd -- Windows thin launcher (double-click / CMD / PowerShell)
rem  Only locates a Python 3.8+ interpreter, then forwards all arguments to
rem  setup_env.py.  All real logic lives in setup_env.py, so Windows and Linux
rem  share exactly one implementation.
rem ============================================================================

setlocal
cd /d "%~dp0"

set "PY_EXE="
set "PY_VER="
set "CHECK=import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"

rem ---- 1) Python launcher: py -3 -------------------------------------------
where py >nul 2>nul
if errorlevel 1 goto :try_python
py -3 -c "%CHECK%" >nul 2>nul
if errorlevel 1 goto :try_python
set "PY_EXE=py"
set "PY_VER=-3"
goto :run

rem ---- 2) python on PATH ---------------------------------------------------
:try_python
where python >nul 2>nul
if errorlevel 1 goto :try_paths
python -c "%CHECK%" >nul 2>nul
if errorlevel 1 goto :try_paths
set "PY_EXE=python"
goto :run

rem ---- 3) common per-user install locations --------------------------------
:try_paths
for %%V in (314 313 312 311 310 39 38) do if not defined PY_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
for %%V in (314 313 312 311 310 39 38) do if not defined PY_EXE if exist "%PROGRAMFILES%\Python%%V\python.exe" set "PY_EXE=%PROGRAMFILES%\Python%%V\python.exe"
for %%V in (314 313 312 311 310 39 38) do if not defined PY_EXE if exist "C:\Python%%V\python.exe" set "PY_EXE=C:\Python%%V\python.exe"
if not defined PY_EXE if exist "%PROGRAMDATA%\Anaconda3\python.exe" set "PY_EXE=%PROGRAMDATA%\Anaconda3\python.exe"
if not defined PY_EXE if exist "%USERPROFILE%\anaconda3\python.exe" set "PY_EXE=%USERPROFILE%\anaconda3\python.exe"
if not defined PY_EXE if exist "%USERPROFILE%\miniconda3\python.exe" set "PY_EXE=%USERPROFILE%\miniconda3\python.exe"

if not defined PY_EXE goto :nopython
"%PY_EXE%" -c "%CHECK%" >nul 2>nul
if errorlevel 1 goto :nopython

rem ---- 4) run ---------------------------------------------------------------
:run
"%PY_EXE%" %PY_VER% "%~dp0setup_env.py" %*
set "RC=%ERRORLEVEL%"

rem Keep the window open when double-clicked (that is, when no argument is given).
if "%~1"=="" pause
exit /b %RC%

rem ---- no python found ------------------------------------------------------
:nopython
echo [fail] No usable Python found on this machine (setup_env.py needs Python 3.8+).
echo.
echo Install options:
echo   1. Install uv first (fastest, and it is this script's preferred manager):
echo        https://docs.astral.sh/uv/getting-started/installation/
echo        then run:  uv python install 3.12
echo                   uv run --python 3.12 python setup_env.py
echo   2. Python from https://www.python.org/downloads/
echo      (tick "Add python.exe to PATH" during installation)
echo   3. Miniconda from https://docs.conda.io/en/latest/miniconda.html
echo.
pause
exit /b 2
