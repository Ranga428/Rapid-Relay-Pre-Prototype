@echo off
REM Always start in the folder where the batch file is located
cd /d "%~dp0"

REM Activate the virtual environment (two levels up from deployment)
call ..\..\floodenv\Scripts\activate.bat

REM Return to the deployment folder (where Start.py is)
cd /d "%~dp0"

REM Run your Python script with the schedule flag
python Start.py --schedule

REM Keep the window open to see output
pause
