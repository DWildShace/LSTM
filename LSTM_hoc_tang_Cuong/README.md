# AI RẮN SĂN MỒI - HỌC TĂNG CƯỜNG (REINFORCEMENT LEARNING)
## MÔ HÌNH DOUBLE DEEP RECURRENT Q-NETWORK (DRQN 2 TẦNG LSTM)

---

## 1. TỔNG QUAN & SO SÁNH PHƯƠNG THỨC HỌC

Trong dự án này, chúng ta xây dựng mô hình AI chơi Rắn săn mồi trên bàn cờ $14 \times 14$ có **4 chướng ngại vật cố định** và **thân rắn là chướng ngại vật động** bằng phương thức **Học Tăng Cường (Reinforcement Learning)**, giữ nguyên $100\%$ kiến trúc nơ-ron từ mô hình [LSTM_hco_giam_sat](file:///e:/________thuMucHoc/ky_5/tri_tue_nhan_tao/tro_choi_ran_san_moi/LSTM_hco_giam_sat).

| Đặc điểm | Học Có Giám Sát (`LSTM_hco_giam_sat`) | Học Tăng Cường (`LSTM_hoc_tang_Cuong`) |
| :--- | :--- | :--- |
| **Nguồn dữ liệu học** | Dữ liệu mẫu bước đi sinh ra từ thuật toán chuyên gia $A^*$ (Imitation Learning). | Không cần chuyên gia $A^*$. Rắn tự tương tác với môi trường, nhận phản hồi thưởng/phạt. |
| **Bản chất đầu ra** | Phân phối xác suất Softmax $[P_{\text{thẳng}}, P_{\text{trái}}, P_{\text{phải}}]$. | Hàm giá trị $Q(s, a)$ dạng Linear ước lượng tổng điểm thưởng kỳ vọng trong tương lai. |
| **Mục tiêu tối ưu** | Khớp hành vi của chuyên gia $A^*$ qua hàm mất mát Cross-Entropy. | Tối đa hóa tổng điểm thưởng tích lũy $\sum_{t} \gamma^t R_t$ qua phương trình Bellman. |
| **Khả năng đột phá** | Bị giới hạn ở mức chuyên gia $A^*$ (khó thông minh hơn dữ liệu mẫu). | Có khả năng tự khám phá các chiến thuật mới tối ưu hơn thông qua thử và sai (Trial-and-Error). |

---

## 2. KIẾN TRÚC MẠNG NƠ-RON (DRQN 2 TẦNG LSTM)

Mô hình kế thừa chính xác kiến trúc các tầng từ mô hình có giám sát, tích hợp **LayerNormalization** để ổn định gradient và hội tụ nhanh:

```text
Chuỗi đầu vào: (batch_size, seq_len=4, feature_dim=16)
                    │
┌───────────────────▼───────────────────┐
│ LayerNormalization (Chuẩn hóa đầu vào) │
└───────────────────┬───────────────────┘
┌───────────────────▼───────────────────┐
│ LSTM Layer 1 (64 units)               │  -> return_sequences=True
└───────────────────┬───────────────────┘
┌───────────────────▼───────────────────┐
│ LayerNormalization (Chuẩn hóa LSTM 1)  │
└───────────────────┬───────────────────┘
┌───────────────────▼───────────────────┐
│ Dropout (0.2)                         │  -> Chống quá khớp (Overfitting)
└───────────────────┬───────────────────┘
┌───────────────────▼───────────────────┐
│ LSTM Layer 2 (32 units)               │  -> return_sequences=False
└───────────────────┬───────────────────┘
┌───────────────────▼───────────────────┐
│ LayerNormalization (Chuẩn hóa LSTM 2)  │
└───────────────────┬───────────────────┘
┌───────────────────▼───────────────────┐
│ Dense Layer (32 units, ReLU)          │  -> Trích xuất đặc trưng phi tuyến
└───────────────────┬───────────────────┘
┌───────────────────▼───────────────────┐
│ Dense Output (3 units, Linear)        │  -> [Q(Đi thẳng), Q(Rẽ trái), Q(Rẽ phải)]
└───────────────────────────────────────┘
```

> [!IMPORTANT]
> **Điểm khác biệt cốt lõi ở tầng ra**: Tầng cuối cùng sử dụng hàm kích hoạt **Linear** (thay vì Softmax) vì các giá trị $Q(s, a)$ biểu thị điểm thưởng kỳ vọng có thể nhận giá trị âm hoặc dương bất kỳ:
> $$Q(s, a) \in \mathbb{R}$$

---

## 3. CƠ CHẾ HỌC TĂNG CƯỜNG (DRQN & BELLMAN EQUATION)

### 3.1. Tại sao dùng LSTM trong Học tăng cường (DRQN)?
Môi trường cờ $14 \times 14$ chứa thân rắn di chuyển liên tục là một bài toán **POMDP (Partially Observable Markov Decision Process)**. Nếu chỉ nhìn $1$ thời điểm $t$, AI không thể phân biệt rắn đang di chuyển theo hướng xoắn ốc hay đang chuẩn bị tự đóng lồng nhốt chính mình. Chuỗi $4$ bước thời gian liên tiếp $(s_{t-3}, s_{t-2}, s_{t-1}, s_t)$ giúp tế bào bộ nhớ LSTM nắm bắt được **động lực học di chuyển và xu hướng co dãn của thân rắn**.

### 3.2. Thuật toán Double Q-Learning
Để tránh hiện tượng ước lượng phóng đại giá trị $Q$ (Overestimation Bias), mạng sử dụng cơ chế **Double Q-Network**:
1. **Policy Network ($Q_{\theta}$)**: Dùng để chọn hành động tối ưu ở trạng thái kế tiếp $s'$:
   $$a^* = \arg\max_{a'} Q_{\theta}(s', a')$$
2. **Target Network ($Q_{\theta^-}$)**: Đánh giá giá trị kỳ vọng của hành động $a^*$ đó:
   $$y = r + \gamma \cdot Q_{\theta^-}(s', a^*)$$
Target Network được đóng băng trọng số và định kỳ đồng bộ từ Policy Network sau mỗi $10$ ván game.

### 3.3. Experience Replay (Bộ nhớ trải nghiệm)
Lưu trữ $50,000$ chuyển trạng thái $(s_t, a_t, r_t, s_{t+1}, \text{done})$. Mỗi bước đi, AI lấy ngẫu nhiên một mini-batch gồm $64$ mẫu để cập nhật trọng số. Việc này giúp **bẻ gãy mối tương quan chuỗi thời gian (Break Temporal Correlation)**, giúp quá trình hội tụ của SGD/Adam diễn ra ổn định.

### 3.4. Định hình phần thưởng (Reward Shaping)
Để rắn học nhanh và tránh đi vòng tròn vô tận:
- **Ăn được mồi**: $+15.0$
- **Va chạm tường / vật cản / cắn thân**: $-15.0$ (Game Over)
- **Tiến lại gần mồi (khoảng cách Manhattan giảm)**: $+0.2$
- **Đi xa khỏi mồi**: $-0.25$
- **Phí thời gian mỗi bước**: $-0.02$ (thúc đẩy tìm đường ngắn nhất)
- **Cơ chế chống đói (Loop Timeout)**: Nếu đi quá $30 \times \text{độ\_dài}$ bước mà không ăn mồi $\rightarrow$ Kết thúc ván và phạt $-10.0$.

---

## 4. CẤU TRÚC THƯ MỤC

```text
LSTM_hoc_tang_Cuong/
│
├── config.py             # Cấu hình siêu tham số RL (Gamma, Epsilon, Batch size, LR...)
├── feature_extractor.py  # Trích xuất 16 đặc trưng hình học & cảm biến tương đối
├── model.py              # Định nghĩa kiến trúc DRQN (2 tầng LSTM + LayerNorm + Dense Linear)
├── environment.py        # Môi trường bọc SnakeGame chuẩn hóa chuỗi thời gian & Reward Shaping
├── agent.py              # SnakeRLAgent: Double DRQN, Replay Buffer, Epsilon-Greedy
├── train_rl.py           # Vòng lặp huấn luyện tự động, lưu checkpoint, báo cáo chỉ số
├── play_ai.py            # Giao diện đồ họa Pygame trực quan hóa Q-values thời gian thực
├── main.py               # Điểm khởi chạy nhanh ứng dụng
├── run_train.bat         # File kích hoạt huấn luyện 1-click trên Windows
├── run_play.bat          # File khởi chạy giao diện xem AI thi đấu 1-click
└── README.md             # Tài liệu chi tiết hướng dẫn và lý thuyết
```

---

## 5. HƯỚNG DẪN CHẠY VÀ SỬ DỤNG

### Bước 1: Huấn luyện mô hình AI (Reinforcement Learning)
Nhấp đúp chuột vào file [run_train.bat](file:///e:/________thuMucHoc/ky_5/tri_tue_nhan_tao/tro_choi_ran_san_moi/LSTM_hoc_tang_Cuong/run_train.bat) hoặc mở Terminal gõ lệnh:
```bash
py -3 train_rl.py --episodes 500 --batch-size 64
```

Các tùy chọn nâng cao:
- `--resume`: Tải mô hình đã lưu để tiếp tục huấn luyện tăng cường (Transfer Learning).
- `--episodes 1000`: Tăng số ván huấn luyện.
- `--batch-size 128`: Tăng kích thước batch khi cập nhật gradient.
- `--lr 0.0003`: Tinh chỉnh tốc độ học.

### Bước 2: Quan sát AI thi đấu với Dashboard Q-Values trực quan
Nhấp đúp chuột vào file [run_play.bat](file:///e:/________thuMucHoc/ky_5/tri_tue_nhan_tao/tro_choi_ran_san_moi/LSTM_hoc_tang_Cuong/run_play.bat) hoặc chạy:
```bash
py -3 play_ai.py
```

### Các phím tắt trong khi xem AI chơi:
- `SPACE`: Tạm dừng / Tiếp tục ván game.
- `R`: Khởi động lại ván chơi mới.
- `MŨI TÊN LÊN (▲)`: Tăng tốc độ game FPS (tối đa $60$ FPS).
- `MŨI TÊN XUỐNG (▼)`: Giảm tốc độ game FPS để quan sát từng bước suy luận Q-value.
- `ESC`: Thoát trò chơi.
