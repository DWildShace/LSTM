"""
Mô hình mạng nơ-ron hồi quy DRQN (Deep Recurrent Q-Network) với khối LSTM.
Kiến trúc bao gồm:
1. Fully Connected Layer (Linear) đầu vào: Ánh xạ vector trạng thái đặc trưng (22 chiều) lên không gian ẩn.
2. Layer Normalization (Chuẩn hoá theo lớp): Giúp ổn định phân phối giá trị đặc trưng, triệt tiêu bão hoà gradient.
3. Activation Layer (ReLU): Hàm kích hoạt phi tuyến tính.
4. LSTM Layer (Khối tế bào bộ nhớ dài-ngắn hạn):
   - Mỗi tế bào là 1 khối (block) gồm 3 cổng cơ bản:
     * Cổng quên (Forget Gate - f_t): Quyết định lượng thông tin cũ trong cell state cần xoá bỏ.
     * Cổng vào (Input Gate - i_t): Quyết định lượng thông tin mới được nạp vào cell state.
     * Cổng ra (Output Gate - o_t): Điều tiết thông tin từ cell state được lọc ra hidden state.
   - Cell state (c_t): Luồng truyền thông tin nội bộ xuyên suốt chuỗi thời gian, bảo tồn bộ nhớ ngữ cảnh dài hạn.
   - Hidden state (h_t): Vừa truyền thông tin nội bộ cho bước thời gian tiếp theo, vừa xuất ra ngoài lớp cho các lớp tiếp theo.
5. Layer Normalization & Activation sau LSTM: Chuẩn hoá và kích hoạt vector biểu diễn ngữ cảnh.
6. Fully Connected Hidden Layer: Nén và kết hợp đặc trưng.
7. Fully Connected Output Layer: Xuất ra 3 giá trị Q-Value cho 3 hành động tương đối: [Đi thẳng, Rẽ trái, Rẽ phải].
"""

import torch
import torch.nn as nn


class SnakeLSTM_QNetwork(nn.Module):
    def __init__(self, input_dim=22, hidden_fc=128, lstm_hidden=128, mid_fc=64, output_dim=3):
        """
        Khởi tạo kiến trúc mô hình DRQN / LSTM:
        :param input_dim: Số đặc trưng đầu vào của trạng thái trò chơi (mặc định 22).
        :param hidden_fc: Số nơ-ron của lớp Fully Connected đầu vào (đề xuất 128).
        :param lstm_hidden: Kích thước vector trạng thái ẩn của LSTM (đề xuất 128).
        :param mid_fc: Số nơ-ron của lớp Fully Connected trung gian (đề xuất 64).
        :param output_dim: Số hành động đầu ra (3 hành động: Đi thẳng, Rẽ trái, Rẽ phải).
        """
        super(SnakeLSTM_QNetwork, self).__init__()

        self.input_dim = input_dim
        self.hidden_fc = hidden_fc
        self.lstm_hidden = lstm_hidden
        self.mid_fc = mid_fc
        self.output_dim = output_dim

        # 1. Lớp Fully Connected đầu vào (Input Projection)
        self.fc_in = nn.Linear(input_dim, hidden_fc)
        # 2. Lớp Layer Normalization chuẩn hoá đầu vào cho LSTM
        self.ln_in = nn.LayerNorm(hidden_fc)
        # 3. Lớp kích hoạt Activation phi tuyến
        self.act_in = nn.ReLU()

        # 4. Lớp đặc trưng LSTM (Recurrent Block với 3 cổng: Forget, Input, Output)
        # nn.LSTM trong PyTorch cài đặt chuẩn:
        # f_t = sigmoid(W_f x_t + U_f h_{t-1} + b_f)       [Cổng Quên]
        # i_t = sigmoid(W_i x_t + U_i h_{t-1} + b_i)       [Cổng Vào]
        # \tilde{c}_t = tanh(W_c x_t + U_c h_{t-1} + b_c)  [Ứng viên Cell State]
        # c_t = f_t * c_{t-1} + i_t * \tilde{c}_t          [Cell State - Truyền nội bộ]
        # o_t = sigmoid(W_o x_t + U_o h_{t-1} + b_o)       [Cổng Ra]
        # h_t = o_t * tanh(c_t)                            [Hidden State - Truyền nội bộ & xuất ra ngoài]
        self.lstm = nn.LSTM(
            input_size=hidden_fc,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        # 5. Layer Normalization & Activation sau LSTM
        self.ln_lstm = nn.LayerNorm(lstm_hidden)
        self.act_lstm = nn.ReLU()

        # 6. Lớp Fully Connected trung gian (Bottleneck Representation)
        self.fc_mid = nn.Linear(lstm_hidden, mid_fc)
        self.ln_mid = nn.LayerNorm(mid_fc)
        self.act_mid = nn.ReLU()

        # 7. Lớp Output Fully Connected (Dự đoán Q-values cho 3 hành động)
        self.fc_out = nn.Linear(mid_fc, output_dim)

    def init_hidden(self, batch_size=1, device="cpu"):
        """
        Khởi tạo trạng thái ẩn h_0 (hidden state) và c_0 (cell state) về 0.
        Kích thước: (num_layers, batch_size, lstm_hidden)
        """
        h_0 = torch.zeros(1, batch_size, self.lstm_hidden, dtype=torch.float32, device=device)
        c_0 = torch.zeros(1, batch_size, self.lstm_hidden, dtype=torch.float32, device=device)
        return (h_0, c_0)

    def forward(self, x, hidden_state=None):
        """
        Lan truyền tiến (Forward Pass):
        :param x: Tensor đầu vào.
                  Có thể là (batch_size, input_dim) cho bước đơn
                  hoặc (batch_size, seq_len, input_dim) cho huấn luyện theo chuỗi.
        :param hidden_state: Bộ đôi (h_t, c_t) của LSTM. Nếu None, sẽ khởi tạo bằng 0.
        :return: (q_values, (h_n, c_n))
        """
        device = x.device
        is_single_step = (x.dim() == 2)

        if is_single_step:
            # Chuyển (batch_size, input_dim) -> (batch_size, 1, input_dim)
            x = x.unsqueeze(1)

        batch_size, seq_len, _ = x.shape

        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size=batch_size, device=device)

        # 1. Chiếu qua lớp Fully Connected đầu vào
        out = self.fc_in(x)
        # 2. Chuẩn hoá theo lớp (LayerNorm)
        out = self.ln_in(out)
        # 3. Kích hoạt phi tuyến (ReLU)
        out = self.act_in(out)

        # 4. Đưa qua khối LSTM
        # out: (batch_size, seq_len, lstm_hidden)
        # next_hidden: (h_n, c_n), mỗi tensor có kích thước (1, batch_size, lstm_hidden)
        lstm_out, next_hidden = self.lstm(out, hidden_state)

        # 5. LayerNorm và ReLU sau LSTM
        lstm_out = self.ln_lstm(lstm_out)
        lstm_out = self.act_lstm(lstm_out)

        # 6. Lớp Fully Connected trung gian
        mid = self.fc_mid(lstm_out)
        mid = self.ln_mid(mid)
        mid = self.act_mid(mid)

        # 7. Tính giá trị Q cho 3 hành động
        q_values = self.fc_out(mid)

        if is_single_step:
            # Ép lại về (batch_size, output_dim)
            q_values = q_values.squeeze(1)

        return q_values, next_hidden
