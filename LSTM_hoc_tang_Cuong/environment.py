"""
Môi trường huấn luyện Học Tăng Cường (Reinforcement Learning Environment)
Bọc (Wrap) logic SnakeGame và chuyển đổi thành giao diện chuẩn MDP (Markov Decision Process):
- Không gian quan sát (State): Chuỗi SEQ_LEN bước quan sát liên tiếp shape (SEQ_LEN, 16)
- Không gian hành động (Action): 3 hành động tương đối: 0: Đi thẳng, 1: Rẽ trái, 2: Rẽ phải
- Hàm phần thưởng (Reward Function): Định hình phần thưởng (Reward Shaping) tối ưu cho RL:
  + Ăn mồi: +15.0
  + Đâm tường / vật cản / cắn đuôi: -15.0
  + Tiến lại gần mồi: +0.2
  + Đi xa mồi: -0.25
  + Phạt thời gian mỗi bước: -0.02 (thúc đẩy đường đi ngắn nhất)
  + Phạt vòng lặp (Loop Penalty): kết thúc ván nếu đi quá lâu không ăn mồi
"""

import sys
import os
from collections import deque
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
ASTAR_DIR = os.path.join(PARENT_DIR, "tim_kiem_A_sao")

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if ASTAR_DIR not in sys.path:
    sys.path.insert(0, ASTAR_DIR)

from constants import (  # noqa: E402
    ACTION_STRAIGHT, ACTION_LEFT, ACTION_RIGHT,
    DEFAULT_NUM_OBSTACLES
)
from snake_game import SnakeGame  # noqa: E402
from feature_extractor import extract_features  # noqa: E402
from config import SEQ_LEN, FEATURE_DIM, NUM_ACTIONS  # noqa: E402


class SnakeRLEnvironment:
    """
    Môi trường RL cho Rắn săn mồi hỗ trợ chuỗi trạng thái đầu vào cho mạng LSTM.
    """
    def __init__(self, num_obstacles: int = DEFAULT_NUM_OBSTACLES, seq_len: int = SEQ_LEN):
        self.num_obstacles = num_obstacles
        self.seq_len = seq_len
        self.feature_dim = FEATURE_DIM
        self.action_space_size = NUM_ACTIONS
        self.game = SnakeGame(num_obstacles=self.num_obstacles)

        # Bộ đệm trượt lưu giữ chuỗi bước thời gian cho LSTM
        self.state_buffer = deque(maxlen=self.seq_len)
        self.steps_without_food = 0

    def reset(self, randomize_obstacles: bool = True) -> np.ndarray:
        """
        Khởi động lại ván chơi mới.
        Trả về chuỗi trạng thái ban đầu có shape (seq_len, feature_dim).
        """
        self.game.reset(randomize_obstacles=randomize_obstacles)
        self.steps_without_food = 0

        # Trích xuất vector 16 đặc trưng đầu tiên
        initial_feat = extract_features(
            self.game.head,
            self.game.direction,
            self.game.body,
            self.game.obstacles,
            self.game.food
        )

        # Đổ đầy buffer bằng vector ban đầu để mạng LSTM có đủ chuỗi đầu vào
        self.state_buffer.clear()
        for _ in range(self.seq_len):
            self.state_buffer.append(initial_feat.copy())

        return np.array(self.state_buffer, dtype=np.float32)

    def _get_manhattan_distance_to_food(self) -> float:
        """Khoảng cách Manhattan từ đầu rắn tới mồi."""
        if self.game.food is None:
            return 0.0
        return abs(self.game.head.x - self.game.food.x) + abs(self.game.head.y - self.game.food.y)

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Thực hiện một bước đi trong môi trường:
        - action_idx: 0 (Đi thẳng), 1 (Rẽ trái), 2 (Rẽ phải)
        
        Trả về tuple: (next_state_seq, reward, done, info)
        """
        # Chuyển đổi chỉ số hành động sang vector one-hot của game
        if action_idx == 0:
            action = ACTION_STRAIGHT
        elif action_idx == 1:
            action = ACTION_LEFT
        elif action_idx == 2:
            action = ACTION_RIGHT
        else:
            raise ValueError(f"Chỉ số hành động không hợp lệ: {action_idx}")

        # Khoảng cách tới mồi trước khi di chuyển
        prev_dist = self._get_manhattan_distance_to_food()

        # Thực thi bước đi trong SnakeGame
        raw_reward, game_over, score = self.game.play_step(action)
        self.steps_without_food += 1

        # Reward Shaping tính toán phần thưởng cho mô hình RL
        reward = 0.0
        if game_over:
            # Phạt nặng khi va chạm gây Game Over
            reward = -15.0
        elif raw_reward > 0:
            # Thưởng lớn khi ăn được mồi
            reward = 15.0
            self.steps_without_food = 0
        else:
            # Rắn vẫn an toàn tiếp tục di chuyển
            new_dist = self._get_manhattan_distance_to_food()
            if new_dist < prev_dist:
                # Thưởng nhẹ khi di chuyển thu hẹp khoảng cách tới mồi
                reward += 0.2
            else:
                # Phạt nhẹ khi di chuyển đi xa khỏi mồi
                reward -= 0.25

            # Phạt chi phí bước đi để tránh đi lang thang
            reward -= 0.02

            # Cơ chế chống lặp vô tận (Loop timeout)
            # Rắn càng dài cho phép tối đa bước đi dài hơn
            max_starve_steps = max(100, 30 * (len(self.game.body) + 1))
            if self.steps_without_food > max_starve_steps:
                game_over = True
                reward = -10.0

        # Trích xuất vector đặc trưng trạng thái mới
        new_feat = extract_features(
            self.game.head,
            self.game.direction,
            self.game.body,
            self.game.obstacles,
            self.game.food
        )

        # Cập nhật buffer chuỗi trạng thái trượt
        self.state_buffer.append(new_feat)
        next_state_seq = np.array(self.state_buffer, dtype=np.float32)

        info = {
            "score": score,
            "steps": self.game.steps,
            "head": (self.game.head.x, self.game.head.y),
            "record": self.game.record_score
        }

        return next_state_seq, reward, game_over, info
