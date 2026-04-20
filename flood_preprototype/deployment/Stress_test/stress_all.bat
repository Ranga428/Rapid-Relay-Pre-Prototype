@echo off
REM Always start in the folder where the batch file is located (deployment\Stress_test\)
cd /d "%~dp0"

REM Activate the virtual environment
call ..\..\..\floodenv\Scripts\activate.bat

REM ── Resolve key paths ────────────────────────────────────────────────────
SET SCRIPTS_DIR=%~dp0..\..\scripts
SET SHOWCASE_DIR=D:\Rapid-Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\showcase
SET STRESS_DIR=D:\Rapid-Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\data\stress_test

echo.
echo ============================================================
echo  Rapid Relay EWS - Stress Test Pipeline [ALL TIERS]
echo  Runs full 3-step pipeline for each tier in sequence:
echo    CLEAR ^> WATCH ^> WARNING ^> DANGER
echo  Step 1 : Generate 30-day stress CSV per tier
echo  Step 2 : Seed showcase_sensor.csv with context (no alert)
echo  Step 3 : Insert 1 live Supabase row (triggers alert)
echo ============================================================
echo.

REM ════════════════════════════════════════════════════════════
REM  CLEAR  (1/4)
REM ════════════════════════════════════════════════════════════
echo ── CLEAR (1/4) ─────────────────────────────────────────────
echo [1/3] Generating 30-day CLEAR stress CSV...
cd /d "%SCRIPTS_DIR%"
python stress_test_gen.py --scenario clear --days 30 --csv
if %ERRORLEVEL% neq 0 ( echo ERROR: CLEAR CSV generation failed. & pause & exit /b 1 )
echo       Done.

echo [2/3] Seeding showcase_sensor.csv with CLEAR context (no alert)...
cd /d "%SHOWCASE_DIR%"
python showcase_predict.py --seed "%STRESS_DIR%\stress_clear.csv"
if %ERRORLEVEL% neq 0 ( echo ERROR: CLEAR seed failed. & pause & exit /b 1 )
echo       Done.

echo [3/3] Inserting 1 live CLEAR row into Supabase (date = day 31 of context)...
cd /d "%SCRIPTS_DIR%"
python stress_test_gen.py --scenario clear --supabase --count 1 --after-csv "%STRESS_DIR%\stress_clear.csv"
if %ERRORLEVEL% neq 0 ( echo ERROR: CLEAR Supabase insert failed. & pause & exit /b 1 )
echo       Done.
echo.

REM ════════════════════════════════════════════════════════════
REM  WATCH  (2/4)
REM ════════════════════════════════════════════════════════════
echo ── WATCH (2/4) ─────────────────────────────────────────────
echo [1/3] Generating 30-day WATCH stress CSV...
cd /d "%SCRIPTS_DIR%"
python stress_test_gen.py --scenario watch --days 30 --csv
if %ERRORLEVEL% neq 0 ( echo ERROR: WATCH CSV generation failed. & pause & exit /b 1 )
echo       Done.

echo [2/3] Seeding showcase_sensor.csv with WATCH context (no alert)...
cd /d "%SHOWCASE_DIR%"
python showcase_predict.py --seed "%STRESS_DIR%\stress_watch.csv"
if %ERRORLEVEL% neq 0 ( echo ERROR: WATCH seed failed. & pause & exit /b 1 )
echo       Done.

echo [3/3] Inserting 1 live WATCH row into Supabase (date = day 31 of context)...
cd /d "%SCRIPTS_DIR%"
python stress_test_gen.py --scenario watch --supabase --count 1 --after-csv "%STRESS_DIR%\stress_watch.csv"
if %ERRORLEVEL% neq 0 ( echo ERROR: WATCH Supabase insert failed. & pause & exit /b 1 )
echo       Done.
echo.

REM ════════════════════════════════════════════════════════════
REM  WARNING  (3/4)
REM ════════════════════════════════════════════════════════════
echo ── WARNING (3/4) ───────────────────────────────────────────
echo [1/3] Generating 30-day WARNING stress CSV...
cd /d "%SCRIPTS_DIR%"
python stress_test_gen.py --scenario warning --days 30 --csv
if %ERRORLEVEL% neq 0 ( echo ERROR: WARNING CSV generation failed. & pause & exit /b 1 )
echo       Done.

echo [2/3] Seeding showcase_sensor.csv with WARNING context (no alert)...
cd /d "%SHOWCASE_DIR%"
python showcase_predict.py --seed "%STRESS_DIR%\stress_warning.csv"
if %ERRORLEVEL% neq 0 ( echo ERROR: WARNING seed failed. & pause & exit /b 1 )
echo       Done.

echo [3/3] Inserting 1 live WARNING row into Supabase (date = day 31 of context)...
cd /d "%SCRIPTS_DIR%"
python stress_test_gen.py --scenario warning --supabase --count 1 --after-csv "%STRESS_DIR%\stress_warning.csv"
if %ERRORLEVEL% neq 0 ( echo ERROR: WARNING Supabase insert failed. & pause & exit /b 1 )
echo       Done.
echo.

REM ════════════════════════════════════════════════════════════
REM  DANGER  (4/4)
REM ════════════════════════════════════════════════════════════
echo ── DANGER (4/4) ────────────────────────────────────────────
echo [1/3] Generating 30-day DANGER stress CSV...
cd /d "%SCRIPTS_DIR%"
python stress_test_gen.py --scenario danger --days 30 --csv
if %ERRORLEVEL% neq 0 ( echo ERROR: DANGER CSV generation failed. & pause & exit /b 1 )
echo       Done.

echo [2/3] Seeding showcase_sensor.csv with DANGER context (no alert)...
cd /d "%SHOWCASE_DIR%"
python showcase_predict.py --seed "%STRESS_DIR%\stress_danger.csv"
if %ERRORLEVEL% neq 0 ( echo ERROR: DANGER seed failed. & pause & exit /b 1 )
echo       Done.

echo [3/3] Inserting 1 live DANGER row into Supabase (date = day 31 of context)...
cd /d "%SCRIPTS_DIR%"
python stress_test_gen.py --scenario danger --supabase --count 1 --after-csv "%STRESS_DIR%\stress_danger.csv"
if %ERRORLEVEL% neq 0 ( echo ERROR: DANGER Supabase insert failed. & pause & exit /b 1 )
echo       Done.
echo.

echo ============================================================
echo  DONE - ALL TIERS stress test pipeline complete.
echo  Each tier: 30-row context seeded + 1 live Supabase row inserted
echo  Live rows : 4 inserted into Supabase (alerts triggered per tier)
echo ============================================================
echo.
pause
