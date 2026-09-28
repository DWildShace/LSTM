@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo ======================================================================
echo     KHỞI ĐỘNG AI RẮN SĂN MỒI - HỌC TĂNG CƯỜNG (DRQN 2 TẦNG LSTM)
echo ======================================================================
echo.
py -3 play_ai.py
pause
