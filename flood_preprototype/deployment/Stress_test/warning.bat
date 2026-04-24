@echo off
REM 1. Go to the exact folder where simulation.py is located
cd /d "D:\Rapid Relay\Rapid-Relay-Pre-Prototype\flood_preprototype\deployment"

REM 2. Activate the virtual environment
call ..\..\floodenv\Scripts\activate.bat

REM 3. Run the simulation for the WARNING tier
python simulation.py --force-alerts --tiers warning

REM Keep the window open
pause