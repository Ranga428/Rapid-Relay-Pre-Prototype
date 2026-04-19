@echo off
REM Always start in the folder where the batch file is located (deployment\Stress_test\)
cd /d "%~dp0"

REM Activate the virtual environment (three levels up: Stress_test -> deployment -> flood_preprototype -> floodenv sibling)
call ..\..\..\floodenv\Scripts\activate.bat

REM Move into the scripts folder where stress_test_gen.py lives
cd /d "%~dp0..\..\scripts"

REM Default count = 1. Override by calling: stress_clear.bat 31
SET COUNT=31
IF NOT "%~1"=="" SET COUNT=%~1

echo.
echo ============================================================
echo  Rapid Relay EWS - Stress Test Supabase Insert [CLEAR]
echo  Count: %COUNT%
echo ============================================================
echo.

python stress_test_gen.py --scenario clear --supabase --count %COUNT%
if %ERRORLEVEL% neq 0 (
    echo ERROR: CLEAR insert failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo  DONE - %COUNT% CLEAR row(s) inserted into Supabase.
echo.
pause
