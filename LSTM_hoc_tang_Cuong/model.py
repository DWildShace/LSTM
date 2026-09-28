"""
Định nghĩa kiến trúc mạng AI Deep Recurrent Q-Network (DRQN) kết hợp 2 tầng LSTM
cho bài toán Học Tăng Cường (Reinforcement Learning) trò chơi Rắn săn mồi.

Kiến trúc lớp (Layer Architecture):
Giữ nguyên 100% cấu trúc các tầng nơ-ron như mô hình học có giám sát:
Input Shape: (batch_size, seq_len, feature_dim) = (batch_size, 4, 16)
      │
┌─────▼────────────────────────┐
│  LayerNormalization          │  -> Chuẩn hóa chuỗi cảm biến đầu vào
└─────┬────────────────────────┘
┌─────▼────────────────────────┐
│  LSTM Layer 1 (64 units)     │  -> return_sequences=True
└─────┬────────────────────────┘
┌─────▼────────────────────────┐
│  LayerNormalization          │  -> Chuẩn hóa hidden states sau LSTM 1
└─────┬────────────────────────┘
┌─────▼────────────────────────┐
│  Dropout (0.2)               │  -> Giảm overfitting
└─────┬────────────────────────┘
┌─────▼────────────────────────┐
│  LSTM Layer 2 (32 units)     │  -> return_sequences=False (tổng hợp chuỗi)
└─────┬────────────────────────┘
┌─────▼────────────────────────┐
│  LayerNormalization          │  -> Chuẩn hóa biểu diễn ngữ cảnh
└─────┬────────────────────────┘
┌─────▼────────────────────────┐
│  Dense Layer (32, ReLU)      │  -> Trích xuất đặc trưng phi tuyến
└─────┬────────────────────────┘
┌─────▼────────────────────────┐
│  Dense Output (3, Linear)    │  -> Ước lượng hàm giá trị Q-value [Q_thẳng, Q_trái, Q_phải]
└──────────────────────────────┘

Lưu ý quan trọng đối với Học tăng cường (Q-Learning):
Tầng Dense Output dùng activation="linear" (thay vì Softmax) vì giá trị Q(s, a) là tổng phần
thưởng kỳ vọng tích lũy có thể âm hoặc dương theo phương trình Bellman:
Q(s, a) = r + gamma * max_a'(Q_target(s', a'))
"""

import sys
import os

# Đảm bảo in tiếng Việt chuẩn trên Windows console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Sử dụng backend torch cho Keras 3
os.environ["KERAS_BACKEND"] = "torch"

from config import SEQ_LEN, FEATURE_DIM, NUM_ACTIONS, LEARNING_RATE


def build_drqn_model(seq_len: int = SEQ_LEN, 
                     feature_dim: int = FEATURE_DIM, 
                     num_actions: int = NUM_ACTIONS,
                     learning_rate: float = LEARNING_RATE,
                     use_layer_norm: bool = True):
    """
    Xây dựng mô hình DRQN (Deep Recurrent Q-Network) với Keras 3.
    Đầu ra là 3 giá trị Q(s, a) tương ứng với 3 toán tử: Đi thẳng, Rẽ trái, Rẽ phải.
    """
    import keras
    from keras import Sequential
    from keras.layers import Input, LSTM, Dropout, Dense, LayerNormalization

    layers = [Input(shape=(seq_len, feature_dim), name="sensor_sequence_input")]

    if use_layer_norm:
        layers.append(LayerNormalization(name="norm_input"))

    layers.append(LSTM(64, return_sequences=True, name="lstm_layer_1"))

    if use_layer_norm:
        layers.append(LayerNormalization(name="norm_lstm1"))

    layers.append(Dropout(0.2, name="dropout_1"))
    layers.append(LSTM(32, return_sequences=False, name="lstm_layer_2"))

    if use_layer_norm:
        layers.append(LayerNormalization(name="norm_lstm2"))

    layers.append(Dense(32, activation="relu", name="dense_features"))
    # Đầu ra Linear biểu diễn trực tiếp Q-values
    layers.append(Dense(num_actions, activation="linear", name="q_values_output"))

    model = Sequential(layers, name="Snake_DRQN_LSTM")

    # Sử dụng hàm mất mát Huber loss (hoặc MSE) chống bùng nổ gradient trong Q-learning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.Huber(),
        metrics=["mae"]
    )
    return model


# Định nghĩa thêm class PyTorch tương đương
try:
    import torch  # noqa: F401
    import torch.nn as nn

    class PyTorchSnakeDRQN(nn.Module):
        def __init__(self, input_dim=FEATURE_DIM, seq_len=SEQ_LEN, num_actions=NUM_ACTIONS):
            super().__init__()
            self.norm_input = nn.LayerNorm(input_dim)
            self.lstm1 = nn.LSTM(input_dim, 64, batch_first=True)
            self.norm_lstm1 = nn.LayerNorm(64)
            self.dropout = nn.Dropout(0.2)
            self.lstm2 = nn.LSTM(64, 32, batch_first=True)
            self.norm_lstm2 = nn.LayerNorm(32)
            self.dense1 = nn.Linear(32, 32)
            self.relu = nn.ReLU()
            self.q_output = nn.Linear(32, num_actions)

        def forward(self, x):
            # x shape: (batch_size, seq_len, input_dim)
            x = self.norm_input(x)
            out, _ = self.lstm1(x)
            out = self.norm_lstm1(out)
            out = self.dropout(out)
            out, _ = self.lstm2(out)
            out = out[:, -1, :]  # Lấy hidden state ở bước thời gian cuối cùng
            out = self.norm_lstm2(out)
            out = self.relu(self.dense1(out))
            q_values = self.q_output(out)
            return q_values
except ImportError:
    PyTorchSnakeDRQN = None


if __name__ == "__main__":
    print("Kiểm tra khởi tạo mô hình DRQN:")
    m = build_drqn_model()
    m.summary()
