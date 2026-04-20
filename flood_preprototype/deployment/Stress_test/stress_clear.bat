@echo off
REM Always start in the folder where the batch file is located (deployment\Stress_test\)
cd /d "%~dp0"

REM Activate the virtual environment
call ..\..\..\floodenv\Scripts\activate.bat

REM ── Resolve key paths ────────────────────────────────────────────────────
SET SCRIPTS_DIR=%~dp0..\..\scripts
SET SHOWCASE_DIR=D:\Rapid-Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\showcase
SET STRESS_CSV=D:\Rapid-Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\data\stress_test\stress_clear.csv

echo.
echo ============================================================
echo  Rapid Relay EWS - Stress Test Pipeline [CLEAR]
echo  Step 1 : Generate 30-day stress CSV
echo  Step 2 : Seed showcase_sensor.csv with context (no alert)
echo  Step 3 : Insert 1 live Supabase row (triggers alert)
echo ============================================================
echo.

REM ── STEP 1: Generate 30-day stress CSV ──────────────────────────────────
echo [1/3] Generating 30-day CLEAR stress CSV...
cd /d "%SCRIPTS_DIR%"
python stress_test_gen.py --scenario clear --days 30 --csv
if %ERRORLEVEL% neq 0 (
    echo ERROR: CSV generation failed.
    pause
    exit /b %ERRORLEVEL%
)
echo       Done. CSV saved to data\stress_test\stress_clear.csv
echo.

REM ── STEP 2: Seed showcase_sensor.csv with 30 rows of context ─────────────
echo [2/3] Seeding showcase_sensor.csv with 30-day context...
echo       NOTE: Writes directly to showcase_sensor.csv — no Supabase,
echo             no prediction, no alert triggered.
cd /d "%SHOWCASE_DIR%"
python showcase_predict.py --seed "%STRESS_CSV%"
if %ERRORLEVEL% neq 0 (
    echo ERROR: Seed step failed.
    pause
    exit /b %ERRORLEVEL%
)
echo       Done. Rolling-window context is now available.
echo.

REM ── STEP 3: Insert 1 live row into Supabase (triggers the real alert) ────
echo [3/3] Inserting 1 live CLEAR row into Supabase (date = day 31 of context)...
echo       NOTE: This row lands on top of the 30-day context and will
echo             produce an accurate tier prediction + trigger the alert.
cd /d "%SCRIPTS_DIR%"
python stress_test_gen.py --scenario clear --supabase --count 1 --after-csv "%STRESS_CSV%"
if %ERRORLEVEL% neq 0 (
    echo ERROR: Supabase insert failed.
    pause
    exit /b %ERRORLEVEL%
)
echo       Done. Live CLEAR row inserted into Supabase.
echo.

echo ============================================================
echo  DONE - CLEAR stress test pipeline complete.
echo  Context seed  : 30 rows written to showcase_sensor.csv
echo  Live row      : Inserted into Supabase (alert triggered)
echo ============================================================
echo.
pause
