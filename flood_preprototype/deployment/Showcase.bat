@echo off
REM Always start in the folder where the batch file is located (deployment\)
cd /d "%~dp0"

REM Activate the virtual environment (two levels up from deployment\)
call ..\..\floodenv\Scripts\activate.bat

REM Move into the scripts folder where Rapid_Relay_Showcase.py lives
cd /d "%~dp0..\showcase"

REM Run the realtime pipeline
python python .\showcase_start.py

REM Keep the window open to see output
pause