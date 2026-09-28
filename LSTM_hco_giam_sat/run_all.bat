@echo off
chcp 65001 > nul
echo ======================================================================
echo   QUY TRÌNH TỰ ĐỘNG TOÀN DIỆN: THU THẬP -> HUẤN LUYỆN -> CHẠY AI LSTM
echo ======================================================================
cd /d "%~dp0"

echo.
echo [BƯỚC 1/3] THU THẬP DỮ LIỆU & TẠO NHÃN TỪ THUẬT TOÁN A* (150 VÁN)...
py -3 -c "import collect_data; collect_data.collect_expert_demonstrations(num_games=150)"
if errorlevel 1 goto error

echo.
echo [BƯỚC 2/3] BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH AI LSTM...
py -3 train.py
if errorlevel 1 goto error

echo.
echo [BƯỚC 3/3] KHỞI ĐỘNG GAME PYGAME ĐỂ QUAN SÁT AI LSTM THI ĐẤU!
py -3 play_ai.py
goto end

:error
echo [!] Đã xảy ra lỗi trong quá trình thực thi!
pause

:end
