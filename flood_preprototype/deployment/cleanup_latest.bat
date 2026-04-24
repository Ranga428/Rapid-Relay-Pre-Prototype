@echo off
setlocal enabledelayedexpansion

:: ============================================================
:: cleanup_latest.bat
:: Purpose:  Reset state by deleting alert JSON files and
::           removing the latest row(s) from sensor/prediction CSVs
:: ============================================================

:: Adjust this base path to match your Windows directory structure
set "BASE=D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype"

:: ── 1. Delete alert channel state files ─────────────────────
echo [INFO]  --- Cleaning alert channel state ---
call :delete_file "%BASE%\alerts\Channels\last_posted.json"
call :delete_file "%BASE%\alerts\Channels\last_telegram_sent.json"

:: ── 2. Trim latest row from sensor CSVs ─────────────────────
echo [INFO]  --- Trimming latest row from sensor CSVs ---
call :delete_last_csv_row "%BASE%\data\sensor\combined_sensor_context.csv"
call :delete_last_csv_row "%BASE%\data\sensor\obando_environmental_data.csv"
call :delete_last_csv_row "%BASE%\data\sensor\obando_sensor_data.csv"

:: ── 3. Trim latest row from prediction CSV ──────────────────
echo [INFO]  --- Trimming latest row from prediction CSV ---
call :delete_last_csv_row "%BASE%\predictions\flood_xgb_sensor_predictions.csv"

echo [INFO]  --- Cleanup complete ---

:: Pause to keep the window open so you can read the output
pause

:: End main script execution before functions
goto :eof


:: ── Helpers ─────────────────────────────────────────────────

:delete_file
set "file=%~1"
if exist "%file%" (
    del /q "%file%"
    echo [INFO]  Deleted: %file%
) else (
    echo [WARN]  Not found ^(skipped^): %file%
)
exit /b

:delete_last_csv_row
set "file=%~1"

if not exist "%file%" (
    echo [WARN]  Not found ^(skipped^): %file%
    exit /b
)

:: Count lines using PowerShell
for /f %%A in ('powershell -NoProfile -Command "(Get-Content -Path '%file%' -ReadCount 0).Count"') do set "total_lines=%%A"

:: Need at least a header + 1 data row
if !total_lines! LSS 2 (
    echo [WARN]  Nothing to remove ^(file has !total_lines! line^(s^)^): %file%
    exit /b
)

:: Back up before modifying
copy /y "%file%" "%file%.bak" >nul
echo [INFO]  Backup created: %file%.bak

:: Drop the last line using a robust PowerShell command (preserves encoding and formatting)
set /a keep=total_lines - 1
powershell -NoProfile -Command "$lines = [System.IO.File]::ReadAllLines('%file%'); [System.IO.File]::WriteAllLines('%file%.tmp', $lines[0..(%keep%-1)])"

:: Replace original file with the trimmed version
move /y "%file%.tmp" "%file%" >nul

echo [INFO]  Removed last row from: %file%  ^(was !total_lines! lines, now !keep!^)
exit /b