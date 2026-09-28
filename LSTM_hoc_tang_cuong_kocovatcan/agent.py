"""
Tác nhân học tăng cường DRQN (Deep Recurrent Q-Network) cho trò chơi Rắn săn mồi.
Quản lý:
1. Replay Buffer lưu trữ chuỗi thời gian (Sequence Replay Buffer) cho LSTM.
2. Chiến lược Epsilon-Greedy cân bằng giữa Khám phá (Exploration) và Khai thác (Exploitation).
3. Cập nhật Double DQN với Target Network nhằm ổn định quá trình học.
4. Cơ chế Train nối tiếp (Checkpointing): Lưu và khôi phục đầy đủ trọng số, optimizer, epsilon, lịch sử điểm số.
"""

import os
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import SnakeLSTM_QNetwork


class SequentialReplayBuffer:
    """
    Bộ nhớ đệm lưu trữ trải nghiệm theo từng tập (Episode).
    Hỗ trợ trích xuất mẫu theo chuỗi con (Sub-sequence) có độ dài `seq_len`
    để mô hình LSTM học được quan hệ phụ thuộc thời gian và ngữ cảnh lịch sử.
    """

    def __init__(self, capacity=2000, seq_len=8):
        self.capacity = capacity
        self.seq_len = seq_len
        self.buffer = deque(maxlen=capacity)

    def push(self, episode):
        """Lưu một tập chơi hoàn chỉnh dưới dạng các mảng NumPy để trích xuất cực nhanh"""
        if len(episode) > 0:
            ep_dict = {
                "states": np.array([t[0] for t in episode], dtype=np.float32),
                "actions": np.array([t[1] for t in episode], dtype=np.int64),
                "rewards": np.array([t[2] for t in episode], dtype=np.float32),
                "next_states": np.array([t[3] for t in episode], dtype=np.float32),
                "dones": np.array([t[4] for t in episode], dtype=bool),
            }
            self.buffer.append(ep_dict)

    def sample(self, batch_size):
        """
        Lấy ngẫu nhiên batch_size chuỗi con có độ dài seq_len.
        Sử dụng mảng NumPy khởi tạo sẵn để trích xuất trực tiếp bằng con trỏ bộ nhớ (Zero-copy).
        """
        sampled_episodes = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        actual_batch_size = len(sampled_episodes)
        feature_dim = sampled_episodes[0]["states"].shape[1]

        batch_states = np.zeros((actual_batch_size, self.seq_len, feature_dim), dtype=np.float32)
        batch_actions = np.zeros((actual_batch_size, self.seq_len), dtype=np.int64)
        batch_rewards = np.zeros((actual_batch_size, self.seq_len), dtype=np.float32)
        batch_next_states = np.zeros((actual_batch_size, self.seq_len, feature_dim), dtype=np.float32)
        batch_dones = np.ones((actual_batch_size, self.seq_len), dtype=bool)

        for i, ep in enumerate(sampled_episodes):
            ep_len = len(ep["actions"])
            if ep_len >= self.seq_len:
                start_idx = random.randint(0, ep_len - self.seq_len)
                end_idx = start_idx + self.seq_len
                batch_states[i] = ep["states"][start_idx:end_idx]
                batch_actions[i] = ep["actions"][start_idx:end_idx]
                batch_rewards[i] = ep["rewards"][start_idx:end_idx]
                batch_next_states[i] = ep["next_states"][start_idx:end_idx]
                batch_dones[i] = ep["dones"][start_idx:end_idx]
            else:
                batch_states[i, :ep_len] = ep["states"]
                batch_actions[i, :ep_len] = ep["actions"]
                batch_rewards[i, :ep_len] = ep["rewards"]
                batch_next_states[i, :ep_len] = ep["next_states"]
                batch_dones[i, :ep_len] = ep["dones"]

        return (
            torch.from_numpy(batch_states),
            torch.from_numpy(batch_actions),
            torch.from_numpy(batch_rewards),
            torch.from_numpy(batch_next_states),
            torch.from_numpy(batch_dones),
        )

    def __len__(self):
        return len(self.buffer)


class SnakeAgent:
    def __init__(
        self,
        input_dim=22,
        hidden_fc=128,
        lstm_hidden=128,
        mid_fc=64,
        output_dim=3,
        lr=0.0005,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.002,
        epsilon_decay=0.995,
        buffer_capacity=2500,
        seq_len=8,
        batch_size=32,
        target_update_freq=200,
        device=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_dim = input_dim
        self.hidden_fc = hidden_fc
        self.lstm_hidden = lstm_hidden
        self.mid_fc = mid_fc
        self.output_dim = output_dim

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.target_update_freq = target_update_freq
        self.steps_done = 0

        # Mạng Policy và Mạng Target (Double DQN)
        self.policy_net = SnakeLSTM_QNetwork(
            input_dim=input_dim,
            hidden_fc=hidden_fc,
            lstm_hidden=lstm_hidden,
            mid_fc=mid_fc,
            output_dim=output_dim,
        ).to(self.device)

        self.target_net = SnakeLSTM_QNetwork(
            input_dim=input_dim,
            hidden_fc=hidden_fc,
            lstm_hidden=lstm_hidden,
            mid_fc=mid_fc,
            output_dim=output_dim,
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss chống bùng nổ gradient

        self.memory = SequentialReplayBuffer(capacity=buffer_capacity, seq_len=seq_len)

        # Trạng thái ẩn của LSTM trong quá trình chơi trực tiếp
        self.hidden_state = None

    def reset_hidden(self):
        """Khởi tạo lại trạng thái ẩn LSTM khi bắt đầu một ván chơi mới"""
        self.hidden_state = self.policy_net.init_hidden(batch_size=1, device=self.device)

    def select_action(self, state, evaluate=False, action_mask=None):
        """
        Lựa chọn hành động theo chiến lược Epsilon-Greedy kết hợp Action Masking (Khám phá an toàn):
        - Khi Khám phá (random): Chỉ chọn ngẫu nhiên trong các hướng AN TOÀN (loại trừ hướng đâm tường/thân).
        - Khi Khai thác (Q-net): Gán giá trị -inf cho các hướng nguy hiểm nếu có ít nhất 1 hướng an toàn.
        """
        # Xác định danh sách các hành động an toàn 1 bước
        if action_mask is not None:
            safe_actions = [i for i, is_safe in enumerate(action_mask) if is_safe]
        else:
            safe_actions = []
            if state[0] < 0.5:
                safe_actions.append(0)  # Đi thẳng an toàn
            if state[2] < 0.5:
                safe_actions.append(1)  # Rẽ trái an toàn
            if state[1] < 0.5:
                safe_actions.append(2)  # Rẽ phải an toàn

        # Nếu bị dồn vào đường cùng (cả 3 hướng đều chết), fallback chấp nhận mọi hành động
        if not safe_actions:
            safe_actions = [0, 1, 2]

        # 1. Khám phá ngẫu nhiên an toàn (Safe Exploration)
        if not evaluate and random.random() < self.epsilon:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                _, self.hidden_state = self.policy_net(state_tensor, self.hidden_state)
            return random.choice(safe_actions)

        # 2. Khai thác dự đoán từ mạng Q kết hợp Action Masking
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values, self.hidden_state = self.policy_net(state_tensor, self.hidden_state)

            # Áp dụng Action Masking vào Q-values
            masked_q = q_values.clone()
            for a in range(self.output_dim):
                if a not in safe_actions:
                    masked_q[0, a] = -1e9

            action = torch.argmax(masked_q, dim=1).item()

        return action

    def train_step(self):
        """
        Thực hiện một bước tối ưu hoá tham số mô hình trên một batch chuỗi kinh nghiệm
        """
        if len(self.memory) < self.batch_size:
            return None

        # Trích xuất một batch chuỗi trải nghiệm
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 1. Dự đoán Q-values cho toàn bộ chuỗi: shape (batch_size, seq_len, output_dim)
        q_values, _ = self.policy_net(states)

        # Lấy giá trị Q tương ứng với các hành động đã thực hiện
        # actions.unsqueeze(-1): (batch_size, seq_len, 1)
        state_action_values = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)

        # 2. Tính giá trị mục tiêu (Target Q-Values) theo cơ chế Double DQN
        with torch.no_grad():
            # Mạng Policy chọn hành động tốt nhất ở trạng thái tiếp theo
            next_q_policy, _ = self.policy_net(next_states)
            best_actions = torch.argmax(next_q_policy, dim=2, keepdim=True)

            # Mạng Target ước tính giá trị Q của hành động tốt nhất đó
            next_q_target, _ = self.target_net(next_states)
            next_state_values = next_q_target.gather(2, best_actions).squeeze(-1)

            # Công thức Bellman: Q_target = r + gamma * max_a' Q(s', a') * (1 - done)
            expected_state_action_values = rewards + (self.gamma * next_state_values * (~dones))

        # 3. Tính toán Loss và cập nhật trọng số
        loss = self.loss_fn(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        # Cắt xén gradient (Gradient Clipping) để ổn định LSTM
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.steps_done += 1

        # Cập nhật mạng Target định kỳ
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Giảm dần epsilon sau mỗi tập chơi để chuyển dần từ khám phá sang khai thác"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def save_checkpoint(
        self,
        filepath,
        episode,
        total_steps,
        best_score,
        score_history=None,
        loss_history=None,
        extra_meta=None,
    ):
        """
        Lưu toàn diện trạng thái phục vụ việc Huấn luyện nối tiếp (Continuous Training):
        - Trọng số mạng Policy & Target
        - Trạng thái bộ tối ưu (Optimizer State)
        - Giá trị Epsilon hiện tại
        - Số tập (Episode) đã trải qua
        - Điểm số kỷ lục và lịch sử
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        checkpoint = {
            "episode": episode,
            "total_steps": total_steps,
            "epsilon": self.epsilon,
            "best_score": best_score,
            "score_history": score_history or [],
            "loss_history": loss_history or [],
            "policy_net_state": self.policy_net.state_dict(),
            "target_net_state": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "hidden_fc": self.hidden_fc,
                "lstm_hidden": self.lstm_hidden,
                "mid_fc": self.mid_fc,
                "output_dim": self.output_dim,
                "gamma": self.gamma,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
            },
            "extra_meta": extra_meta or {},
        }
        torch.save(checkpoint, filepath)
        print(f"[CHECKPOINT] Đã lưu thành công tại: {filepath} (Episode {episode}, Best Score {best_score})")

    def load_checkpoint(self, filepath, reset_epsilon=None):
        """
        Nạp lại trạng thái đã lưu để huấn luyện nối tiếp hoặc chạy kiểm thử.
        :param reset_epsilon: Nếu cung cấp float (vd 0.2), sẽ thiết lập lại epsilon để tăng cường khám phá thêm.
        :return: (episode, total_steps, best_score, score_history, loss_history)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file checkpoint: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_net_state"])
        self.target_net.load_state_dict(checkpoint["target_net_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        if reset_epsilon is not None:
            self.epsilon = float(reset_epsilon)
        else:
            self.epsilon = float(checkpoint.get("epsilon", self.epsilon_min))

        episode = checkpoint.get("episode", 0)
        total_steps = checkpoint.get("total_steps", 0)
        best_score = checkpoint.get("best_score", 0)
        score_history = checkpoint.get("score_history", [])
        loss_history = checkpoint.get("loss_history", [])

        print(
            f"[RESUME] Đã nạp checkpoint từ: {filepath}\n"
            f"         -> Tiếp tục từ Episode: {episode}, Điểm kỷ lục: {best_score}, Epsilon: {self.epsilon:.4f}"
        )

        return episode, total_steps, best_score, score_history, loss_history
