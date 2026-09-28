@echo off
chcp 65001 > nul
echo ========================================================
echo   CHẠY GAME RẮN SĂN MỒI VỚI MÔ HÌNH AI LSTM
echo ========================================================
cd /d "%~dp0"
py -3 play_ai.py
pause
