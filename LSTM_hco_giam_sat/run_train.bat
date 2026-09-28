@echo off
chcp 65001 > nul
echo ======================================================================
echo             HUẤN LUYỆN MÔ HÌNH AI LSTM CHƠI RẮN SĂN MỒI
echo ======================================================================
echo.
echo Chọn chế độ huấn luyện:
echo   [1] Huấn luyện NỐI TIẾP từ mô hình cũ (Resume Training) [Khuyên dùng]
echo   [2] Khởi tạo ngẫu nhiên và huấn luyện MỚI TỪ ĐẦU
set /p MODE="Chọn chế độ (1 hoặc 2) [Nhấn Enter để chọn 1]: "
if "%MODE%"=="" set MODE=1

echo.
set /p EPOCHS="Nhập số Epochs huấn luyện [Nhấn Enter để chọn mặc định 40]: "
if "%EPOCHS%"=="" set EPOCHS=40

cd /d "%~dp0"
echo.
if "%MODE%"=="1" (
    echo [*] Bắt đầu huấn luyện NỐI TIẾP (Resume) với %EPOCHS% Epochs, Batch size = 128...
    py -3 train.py --epochs %EPOCHS% --batch-size 128 --resume
) else (
    echo [*] Bắt đầu huấn luyện MỚI TỪ ĐẦU với %EPOCHS% Epochs, Batch size = 128...
    py -3 train.py --epochs %EPOCHS% --batch-size 128
)

echo.
echo ======================================================================
echo ĐÃ HOÀN TẤT HUẤN LUYỆN! Bạn hãy chạy 'run_play.bat' để xem AI thi đấu.
echo ======================================================================
pause
