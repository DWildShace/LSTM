"""
Lớp Agent Học Tăng Cường (Reinforcement Learning Agent - DRQN) cho game Rắn săn mồi.
Sử dụng giải thuật Double Deep Recurrent Q-Network (Double DRQN):
- Policy Network: Mạng nơ-ron chính quyết định hành động và liên tục được cập nhật trọng số.
- Target Network: Mạng nơ-ron mục tiêu giữ ổn định hàm giá trị Q, định kỳ đồng bộ trọng số.
- Replay Buffer (Bộ nhớ trải nghiệm): Lưu trữ và lấy mẫu ngẫu nhiên các chuyển trạng thái (s, a, r, s', done).
- Epsilon-Greedy Strategy: Cân bằng giữa khám phá (Exploration) và khai thác (Exploitation).
"""

import sys
import os
import random
from collections import deque
import numpy as np

# Đảm bảo in tiếng Việt chuẩn trên Windows console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Sử dụng backend torch cho Keras 3
os.environ["KERAS_BACKEND"] = "torch"
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import (  # noqa: E402
    SEQ_LEN, FEATURE_DIM, NUM_ACTIONS,
    GAMMA, EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
    MEMORY_SIZE, BATCH_SIZE, LEARNING_RATE,
    MODEL_PATH, MODEL_H5_PATH
)
from model import build_drqn_model  # noqa: E402


class SnakeRLAgent:
    """
    Agent DRQN kết hợp 2 tầng LSTM chơi Rắn săn mồi.
    """
    def __init__(self,
                 seq_len: int = SEQ_LEN,
                 feature_dim: int = FEATURE_DIM,
                 num_actions: int = NUM_ACTIONS,
                 gamma: float = GAMMA,
                 epsilon: float = EPSILON_START,
                 epsilon_min: float = EPSILON_MIN,
                 epsilon_decay: float = EPSILON_DECAY,
                 learning_rate: float = LEARNING_RATE,
                 memory_size: int = MEMORY_SIZE,
                 batch_size: int = BATCH_SIZE):
        
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # 1. Bộ nhớ trải nghiệm (Experience Replay Buffer)
        self.memory = deque(maxlen=memory_size)

        # 2. Xây dựng Mạng Chính (Policy Network) và Mạng Mục Tiêu (Target Network)
        print("[*] Đang khởi tạo Mạng Chính (Policy Network) DRQN...")
        self.policy_net = build_drqn_model(
            seq_len=self.seq_len,
            feature_dim=self.feature_dim,
            num_actions=self.num_actions,
            learning_rate=self.learning_rate
        )

        print("[*] Đang khởi tạo Mạng Mục Tiêu (Target Network) DRQN...")
        self.target_net = build_drqn_model(
            seq_len=self.seq_len,
            feature_dim=self.feature_dim,
            num_actions=self.num_actions,
            learning_rate=self.learning_rate
        )

        # Đồng bộ trọng số ban đầu sang Target Network
        self.update_target_network()

    def update_target_network(self):
        """Sao chép trọng số từ Policy Network sang Target Network."""
        self.target_net.set_weights(self.policy_net.get_weights())

    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """
        Lưu trữ một bước chuyển tiếp vào Replay Buffer:
        (S_t, A_t, R_t, S_{t+1}, Done)
        """
        self.memory.append((state, action, reward, next_state, done))

    def get_q_values(self, state_seq: np.ndarray) -> np.ndarray:
        """
        Dự đoán giá trị Q-values cho 3 hành động [Đi thẳng, Rẽ trái, Rẽ phải]
        Đảm bảo không tính gradient và tối ưu tốc độ suy luận.
        """
        if state_seq.ndim == 2:
            state_input = np.expand_dims(state_seq, axis=0)  # Shape: (1, seq_len, feature_dim)
        else:
            state_input = state_seq

        with torch.no_grad():
            preds = self.policy_net(state_input, training=False)
            if hasattr(preds, 'detach'):
                q_vals = preds.detach().cpu().numpy()[0]
            else:
                q_vals = np.array(preds)[0]
        return q_vals

    def act(self, state_seq: np.ndarray, evaluate: bool = False) -> tuple[int, np.ndarray]:
        """
        Chọn hành động theo chiến lược Epsilon-Greedy:
        - Với xác suất epsilon (khám phá): chọn hành động ngẫu nhiên.
        - Với xác suất 1 - epsilon (khai thác): chọn hành động có Q-value lớn nhất.
        
        Trả về: (action_idx, q_values)
        """
        q_vals = self.get_q_values(state_seq)

        if not evaluate and random.random() < self.epsilon:
            # Khám phá ngẫu nhiên
            action_idx = random.randrange(self.num_actions)
        else:
            # Khai thác: chọn hành động có Q lớn nhất
            action_idx = int(np.argmax(q_vals))

        return action_idx, q_vals

    def replay(self) -> float:
        """
        Lấy mẫu mini-batch từ Replay Buffer và cập nhật trọng số Policy Network
        theo công thức Double Q-Learning:
        Q_target = r + gamma * Q_target(s', argmax_a(Q_policy(s', a)))
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        # Lấy mẫu ngẫu nhiên mini-batch
        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([m[0] for m in minibatch], dtype=np.float32)
        actions = np.array([m[1] for m in minibatch], dtype=np.int32)
        rewards = np.array([m[2] for m in minibatch], dtype=np.float32)
        next_states = np.array([m[3] for m in minibatch], dtype=np.float32)
        dones = np.array([m[4] for m in minibatch], dtype=bool)

        with torch.no_grad():
            # 1. Dự đoán Q-values hiện tại cho states
            current_q = self.policy_net(states, training=False)
            if hasattr(current_q, 'detach'):
                targets = current_q.detach().cpu().numpy().copy()
            else:
                targets = np.array(current_q).copy()

            # 2. Double DQN: Dùng Policy Network chọn hành động tốt nhất ở next_states
            next_q_policy = self.policy_net(next_states, training=False)
            if hasattr(next_q_policy, 'detach'):
                next_q_policy = next_q_policy.detach().cpu().numpy()
            else:
                next_q_policy = np.array(next_q_policy)
            best_next_actions = np.argmax(next_q_policy, axis=1)

            # 3. Dùng Target Network để định lượng giá trị của hành động tốt nhất đó
            next_q_target = self.target_net(next_states, training=False)
            if hasattr(next_q_target, 'detach'):
                next_q_target = next_q_target.detach().cpu().numpy()
            else:
                next_q_target = np.array(next_q_target)

        # 4. Cập nhật Q target theo phương trình Bellman
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                best_act = best_next_actions[i]
                targets[i, actions[i]] = rewards[i] + self.gamma * next_q_target[i, best_act]

        # 5. Huấn luyện 1 batch trên Policy Network
        loss = self.policy_net.train_on_batch(states, targets)
        if isinstance(loss, (list, tuple, np.ndarray)):
            loss = float(loss[0])
        else:
            loss = float(loss)

        return loss

    def decay_epsilon(self):
        """Giảm dần xác suất khám phá epsilon sau mỗi ván game."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

    def save(self, file_path: str = MODEL_PATH):
        """Lưu trọng số mô hình Policy Network."""
        try:
            self.policy_net.save(file_path)
            print(f"[✓] Đã lưu mô hình Policy Network tại: {file_path}")
        except Exception as e:
            print(f"[!] Lỗi lưu {file_path}: {e}")
            if file_path.endswith(".keras"):
                fallback = file_path.replace(".keras", ".h5")
                try:
                    self.policy_net.save(fallback)
                    print(f"[✓] Đã lưu dự phòng tại: {fallback}")
                except Exception as e2:
                    print(f"[!] Lỗi lưu dự phòng: {e2}")

    def load(self, file_path: str = MODEL_PATH) -> bool:
        """Tải mô hình đã huấn luyện từ trước."""
        import keras
        loaded = False
        target_path = file_path
        if not os.path.exists(target_path):
            if os.path.exists(MODEL_H5_PATH):
                target_path = MODEL_H5_PATH

        if os.path.exists(target_path):
            try:
                self.policy_net = keras.models.load_model(target_path)
                self.update_target_network()
                print(f"[✓] Đã tải thành công mô hình từ: {target_path}")
                loaded = True
            except Exception as e:
                print(f"[!] Lỗi tải mô hình {target_path}: {e}")
        else:
            print(f"[*] Chưa tìm thấy file mô hình sẵn có tại {file_path}. Sẽ bắt đầu huấn luyện từ đầu.")
        return loaded
