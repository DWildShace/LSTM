"""
Định nghĩa kiến trúc mạng AI LSTM chơi Rắn săn mồi theo đúng thiết kế:

Input Shape: (batch_size, seq_len, feature_dim)
      │
┌─────▼────────────────────────┐
│  LSTM Layer 1 (64 units)     │  -> return_sequences=True
└─────┬────────────────────────┘
┌─────▼────────────────────────┐
│  Dropout (0.2)               │
└─────┬────────────────────────┘
┌─────▼────────────────────────┐
│  LSTM Layer 2 (32 units)     │  -> return_sequences=False
└─────┬────────────────────────┘
┌─────▼────────────────────────┐
│  Dense Layer (32, ReLU)      │
└─────┬────────────────────────┘
┌─────▼────────────────────────┐
│  Dense Output (3, Softmax)   │  -> Phân phối xác suất [Thẳng, Trái, Phải]
└──────────────────────────────┘
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

from config import SEQ_LEN, FEATURE_DIM, NUM_ACTIONS


def build_keras_model(seq_len: int = SEQ_LEN, 
                      feature_dim: int = FEATURE_DIM, 
                      num_actions: int = NUM_ACTIONS,
                      use_layer_norm: bool = True):
    """
    Xây dựng mô hình AI LSTM chơi Rắn săn mồi.
    Tích hợp LayerNormalization để tăng tốc hội tụ và ổn định gradient.
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
    layers.append(Dense(num_actions, activation="softmax", name="action_probabilities"))

    model = Sequential(layers, name="Snake_AI_LSTM")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# Định nghĩa thêm class PyTorch tương đương dự phòng khi cần chạy thuần PyTorch
try:
    import torch  # noqa: F401
    import torch.nn as nn

    class PyTorchSnakeLSTM(nn.Module):
        def __init__(self, input_dim=FEATURE_DIM, seq_len=SEQ_LEN, num_actions=NUM_ACTIONS):
            super().__init__()
            self.lstm1 = nn.LSTM(input_dim, 64, batch_first=True)
            self.dropout = nn.Dropout(0.2)
            self.lstm2 = nn.LSTM(64, 32, batch_first=True)
            self.dense1 = nn.Linear(32, 32)
            self.relu = nn.ReLU()
            self.dense2 = nn.Linear(32, num_actions)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            # x: (batch_size, seq_len, input_dim)
            out, _ = self.lstm1(x)         # (batch_size, seq_len, 64)
            out = self.dropout(out)
            out, _ = self.lstm2(out)         # (batch_size, seq_len, 32)
            out = out[:, -1, :]            # Lấy bước thời gian cuối cùng: (batch_size, 32)
            out = self.relu(self.dense1(out)) # (batch_size, 32)
            out = self.softmax(self.dense2(out)) # (batch_size, 3)
            return out
except ImportError:
    PyTorchSnakeLSTM = None
