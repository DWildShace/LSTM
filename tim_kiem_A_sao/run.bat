@echo off
chcp 65001 > nul
echo ========================================================
echo   AI SNAKE - TRÒ CHƠI RẮN SĂN MỒI VỚI THUẬT TOÁN A*
echo   Bản đồ 14x14 | 3-4 Vật cản cố định + Thân rắn động
echo ========================================================
echo.
py -3 main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Thử lại với python thông thường...
    python main.py
)
pause
