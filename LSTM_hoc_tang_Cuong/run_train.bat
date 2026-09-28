@echo off
chcp 65001 > nul
echo ======================================================================
echo       HUẤN LUYỆN AI RẮN SĂN MỒI - HỌC TĂNG CƯỜNG (DRQN 2 TẦNG LSTM)
echo ======================================================================
echo.
echo Chọn chế độ huấn luyện:
echo   [1] Huấn luyện NỐI TIẾP từ mô hình cũ (Resume Training) [Khuyên dùng]
echo   [2] Khởi tạo ngẫu nhiên và huấn luyện MỚI TỪ ĐẦU
set /p MODE="Chọn chế độ (1 hoặc 2) [Nhấn Enter để chọn 1]: "
if "%MODE%"=="" set MODE=1

echo.
set /p EPISODES="Nhập số ván game huấn luyện [Nhấn Enter để chọn mặc định 500]: "
if "%EPISODES%"=="" set EPISODES=500

echo.
set /p BATCH="Nhập kích thước batch replay [Nhấn Enter để chọn mặc định 64]: "
if "%BATCH%"=="" set BATCH=64

cd /d "%~dp0"
echo.
if "%MODE%"=="1" (
    echo [*] Bắt đầu huấn luyện NỐI TIẾP (Resume) %EPISODES% ván, Batch = %BATCH%...
    py -3 train_rl.py --episodes %EPISODES% --batch-size %BATCH% --resume
) else (
    echo [*] Bắt đầu huấn luyện MỚI TỪ ĐẦU %EPISODES% ván, Batch = %BATCH%...
    py -3 train_rl.py --episodes %EPISODES% --batch-size %BATCH%
)

echo.
echo ======================================================================
echo ĐÃ HOÀN TẤT HUẤN LUYỆN! Bạn hãy chạy 'run_play.bat' để xem AI thi đấu.
echo ======================================================================
pause
