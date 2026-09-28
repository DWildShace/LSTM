"""
Các cấu hình tham số cho mô hình AI Học Tăng Cường (Reinforcement Learning - DRQN)
sử dụng mạng Deep Recurrent Q-Network kết hợp 2 tầng LSTM.
"""

import os

# Kích thước chuỗi thời gian đưa vào LSTM (Số bước quan sát liên tiếp)
SEQ_LEN = 4

# Số chiều vector đặc trưng trạng thái tại mỗi bước
FEATURE_DIM = 16

# Số lượng hành động tương đối (Action Space): 0: Đi thẳng, 1: Rẽ trái, 2: Rẽ phải
NUM_ACTIONS = 3

# Đường dẫn thư mục và lưu trữ mô hình
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
MODEL_PATH = os.path.join(CURRENT_DIR, "snake_rl_lstm_model.keras")
MODEL_H5_PATH = os.path.join(CURRENT_DIR, "snake_rl_lstm_model.h5")

# Tham số Học Tăng Cường (Reinforcement Learning Hyperparameters)
GAMMA = 0.95                 # Hệ số chiết khấu phần thưởng tương lai (Discount Factor)
EPSILON_START = 1.0          # Xác suất khám phá ban đầu (100% ngẫu nhiên)
EPSILON_MIN = 0.02           # Xác suất khám phá tối thiểu (2% để duy trì thích ứng)
EPSILON_DECAY = 0.996        # Tốc độ giảm dần khám phá sau mỗi ván game (Decay Rate)

MEMORY_SIZE = 50_000         # Kích thước bộ đệm trải nghiệm (Replay Buffer)
BATCH_SIZE = 64              # Kích thước mini-batch khi huấn luyện lại từ bộ đệm
LEARNING_RATE = 0.0005       # Tốc độ học của bộ tối ưu Adam trong RL
TARGET_UPDATE_GAMES = 10     # Số ván cập nhật Target Network một lần để ổn định Q-learning

# Tham số ván game RL
TOTAL_EPISODES = 500         # Tổng số ván huấn luyện tự động
MAX_STEPS_PER_EPISODE = 2000 # Số bước tối đa mỗi ván tránh lặp vô tận
