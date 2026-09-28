@echo off
chcp 65001 > nul
echo ======================================================================
echo          THU THẬP DỮ LIỆU & TẠO NHÃN TỪ THUẬT TOÁN A*
echo ======================================================================
echo.
echo Bạn có thể nhập số ván A* muốn mô phỏng để tạo nhãn.
echo Gợi ý:
echo   - 300 ván  : ~400.000 mẫu  (khoảng 1 phút)
echo   - 500 ván  : ~700.000 mẫu  (khoảng 1.5 - 2 phút)  [Khuyên dùng]
echo   - 1000 ván : ~1.400.000 mẫu (khoảng 3.5 phút)      [Chất lượng cao nhất]
echo.
set /p GAMES="Nhập số ván cần tạo nhãn [Nhấn Enter để chọn mặc định 500]: "
if "%GAMES%"=="" set GAMES=500

echo.
echo Chế độ lưu dữ liệu:
echo   [1] Ghi mới hoàn toàn (Overwrite) - Mặc định
echo   [2] Gộp thêm vào dữ liệu cũ (Append - Tích lũy thêm mẫu)
set /p MODE="Chọn chế độ (1 hoặc 2) [Mặc định 1]: "

cd /d "%~dp0"
echo.
if "%MODE%"=="2" (
    echo [*] Đang chạy thu thập %GAMES% ván và GỘP THÊM vào tập dữ liệu cũ...
    py -3 collect_data.py --games %GAMES% --append
) else (
    echo [*] Đang chạy thu thập %GAMES% ván và GHI MỚI tập dữ liệu...
    py -3 collect_data.py --games %GAMES%
)

echo.
echo ======================================================================
echo ĐÃ HOÀN TẤT THU THẬP! Bạn hãy chạy 'run_train.bat' để huấn luyện lại.
echo ======================================================================
pause
