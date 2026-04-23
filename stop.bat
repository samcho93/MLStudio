@echo off
echo Stopping ML Node Studio...
taskkill /fi "WINDOWTITLE eq ML-Backend*" /f >nul 2>&1
taskkill /fi "WINDOWTITLE eq ML-Frontend*" /f >nul 2>&1
echo All services stopped.
pause
