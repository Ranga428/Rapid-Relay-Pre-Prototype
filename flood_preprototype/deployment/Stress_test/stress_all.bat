@echo off
REM Always start in the folder where the batch file is located (deployment\Stress_test\)
cd /d "%~dp0"

REM Activate the virtual environment (three levels up: Stress_test -> deployment -> flood_preprototype -> floodenv sibling)
call ..\..\..\floodenv\Scripts\activate.bat

REM Move into the scripts folder where stress_test_gen.py lives
cd /d "%~dp0..\..\scripts"

REM Default count = 1. Override by calling: stress_all.bat 31
REM Note: --scenario all inserts 4 rows per count (one per tier)
REM       so stress_all.bat 31 = 124 total rows
SET COUNT=1
IF NOT "%~1"=="" SET COUNT=%~1

echo.
echo ============================================================
echo  Rapid Relay EWS - Stress Test Supabase Insert [ALL]
echo  Count: %COUNT% per tier  (x4 tiers = %COUNT%*4 total rows)
echo ============================================================
echo.

python stress_test_gen.py --scenario all --supabase --count %COUNT%
if %ERRORLEVEL% neq 0 (
    echo ERROR: ALL insert failed.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo  DONE - %COUNT% row(s) per tier inserted into Supabase.
echo.
pause
