"""
Các cấu hình tham số cho mô hình AI LSTM chơi Rắn săn mồi.
"""

import os

# Kích thước chuỗi thời gian đưa vào LSTM (Số bước quan sát liên tiếp)
SEQ_LEN = 8

# Số chiều vector đặc trưng trạng thái tại mỗi bước
# 3 cảm biến nguy hiểm + 4 hướng đi + 4 hướng mồi + 2 khoảng cách mồi + 3 khoảng cách vật cản = 16
FEATURE_DIM = 16

# Số lượng hành động (Toán tử di chuyển): 0: Thẳng, 1: Trái, 2: Phải
NUM_ACTIONS = 3

# Đường dẫn lưu trữ dữ liệu và mô hình
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(CURRENT_DIR, "snake_lstm_dataset.npz")
MODEL_PATH = os.path.join(CURRENT_DIR, "snake_lstm_model.keras")
MODEL_H5_PATH = os.path.join(CURRENT_DIR, "snake_lstm_model.h5")

# Cấu hình thu thập dữ liệu từ A*
NUM_COLLECT_GAMES = 500   # Mặc định 500 ván (khoảng 700.000 mẫu dữ liệu chất lượng cao)
MAX_STEPS_PER_GAME = 2500 # Giới hạn số bước tối đa mỗi ván tránh lặp vô tận

# Cấu hình huấn luyện
BATCH_SIZE = 128          # Tăng batch size lên 128 để tối ưu tốc độ và giảm dao động loss
EPOCHS = 40               # Số epoch tối đa
LEARNING_RATE = 0.001     # Tốc độ học ban đầu của Adam
TEST_SPLIT = 0.15         # 15% dữ liệu dùng để validation kiểm tra chất lượng
