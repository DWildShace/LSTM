# RẮN SĂN MỒI AI - DRQN (LSTM & HỌC TĂNG CƯỜNG)

Dự án xây dựng mô hình AI tự chơi trò **Rắn săn mồi (Snake Game)** trên bản đồ kích thước **14x14 ô không có vật cản**, sử dụng kiến trúc mạng nơ-ron hồi quy **LSTM** kết hợp kỹ thuật **Học tăng cường (Deep Recurrent Q-Network - DRQN / Double DQN)**.

---

## 1. Cấu Trúc Thư Mục
```text
LSTM_hoc_tang_cuong_kocovatcan/
├── snake_env.py        # Môi trường game 14x14, trích xuất 22 features, rendering Pygame
├── model.py            # Mạng DRQN (FC -> LayerNorm -> ReLU -> LSTM -> LayerNorm -> FC -> Output)
├── agent.py            # Tác nhân DRQN, Sequential Replay Buffer, Double DQN, Checkpoint Save/Load
├── train.py            # Kịch bản huấn luyện, hỗ trợ train nối tiếp (--resume), vẽ biểu đồ
├── play.py             # Kịch bản chạy trực quan hóa mô hình đã huấn luyện bằng Pygame
├── requirements.txt    # Danh sách thư viện phụ thuộc
└── checkpoints/        # Thư mục lưu mô hình (latest_checkpoint.pth, best_model.pth, đồ thị)
```

---

## 2. Thiết Kế Trạng Thái Đầu Vào (State - 22 Đặc trưng)
Thay vì sử dụng ảnh thô chiếm nhiều tài nguyên, hệ thống trích xuất **vector đặc trưng 22 chiều** chuẩn hoá trong khoảng `[0.0, 1.0]` hoặc `[-1.0, 1.0]`:

1. **Nguy hiểm 1 bước trực tiếp (3 đặc trưng - Boolean)**:
   - Đi thẳng gặp vật cản (tường hoặc thân)
   - Rẽ phải gặp vật cản
   - Rẽ trái gặp vật cản
2. **Nguy hiểm 2 bước nhìn xa (3 đặc trưng - Boolean)**:
   - Đi thẳng 2 bước gặp vật cản
   - Rẽ phải 2 bước gặp vật cản
   - Rẽ trái 2 bước gặp vật cản
3. **Hướng di chuyển toàn cục hiện tại (4 đặc trưng - One-hot)**:
   - Đang đi LÊN, SANG PHẢI, XUỐNG DƯỚI, SANG TRÁI.
4. **Vị trí thức ăn tương đối theo đầu rắn (4 đặc trưng - Boolean)**:
   - Thức ăn ở phía TRƯỚC
   - Thức ăn ở bên PHẢI
   - Thức ăn ở bên TRÁI
   - Thức ăn ở phía SAU LƯNG
5. **Khoảng cách toạ độ tới thức ăn (3 đặc trưng - Liên tục)**:
   - $\Delta X = (food\_x - head\_x) / 14.0$
   - $\Delta Y = (food\_y - head\_y) / 14.0$
   - Khoảng cách Euclid chuẩn hoá tới mồi.
6. **Tỉ lệ chiều dài thân rắn (1 đặc trưng)**:
   - $len(snake) / 196.0$ (với bản đồ 14x14 có tổng cộng 196 ô).
7. **Khoảng cách tia chiếu (Raycasting) tới chướng ngại vật (3 đặc trưng)**:
   - Đo khoảng cách từ đầu rắn tới tường hoặc thân theo hướng Thẳng, Phải, Trái.
8. **Tỉ lệ không gian an toàn (Flood-Fill Ratio - 1 đặc trưng)**:
   - Thuật toán BFS đếm số ô trống có thể đi tiếp nếu đi thẳng, giúp AI không đi vào ngõ cụt do chính thân rắn cuộn lại.

---

## 3. Không Gian Hành Động Đầu Ra (Action - 3 Trạng thái)
Mô hình đưa ra quyết định dựa trên hệ quy chiếu tương đối của đầu rắn:
- `0`: **Đi thẳng** (giữ nguyên hướng di chuyển hiện tại).
- `1`: **Rẽ trái** (quay 90 độ ngược chiều kim đồng hồ so với hướng hiện tại).
- `2`: **Rẽ phải** (quay 90 độ cùng chiều kim đồng hồ so với hướng hiện tại).

*Lợi thế*: Không gian hành động tương đối loại trừ hoàn toàn việc rắn tự quay ngược 180 độ đâm vào cổ mình, giúp mô hình học hội tụ nhanh hơn nhiều lần so với 4 hướng tuyệt đối.

### 3.1. Kỹ Thuật Action Masking (Khám Phá An Toàn - Safe Exploration)
- **Cơ chế**: Trước khi chọn hành động (dù là ngẫu nhiên hay dự đoán từ mạng Q), môi trường sẽ tính toán `action_mask` để phát hiện các hướng đâm thẳng vào tường hoặc thân rắn trong 1 bước.
- **Khi Khám phá ngẫu nhiên ($\epsilon$-greedy)**: AI chỉ tung xúc xắc giữa các hướng AN TOÀN, loại bỏ $100\%$ các cái chết ngớ ngẩn do random gây ra.
- **Khi Khai thác (Inference)**: Gán giá trị $Q = -\infty$ cho các hành động tự sát nếu còn ít nhất 1 lối thoát an toàn.
- **Kết quả**: Điểm số không bị sụt giảm bất thường, rắn sống dai hơn và quá trình huấn luyện đạt độ ổn định vượt trội.

---

## 4. Thảo Luận & Đề Xuất Số Lượng Nơ-ron Trong Các Lớp

### 4.1. Kiến Trúc Chi Tiết
```text
Input (22 chiều)
     │
     ▼
[Fully Connected Layer: 22 -> 128]
     │
     ▼
[Layer Normalization (128)]
     │
     ▼
[Activation Layer: ReLU]
     │
     ▼
[LSTM Block Layer: 128 -> 128]
     ├── Cổng quên (Forget Gate: f_t)
     ├── Cổng vào  (Input Gate: i_t & c̃_t)
     ├── Cell State (c_t: Truyền nội bộ qua các bước thời gian)
     └── Cổng ra   (Output Gate: o_t -> Hidden State h_t)
     │
     ▼
[Layer Normalization (128)]
     │
     ▼
[Activation Layer: ReLU]
     │
     ▼
[Fully Connected Layer (Mid Bottleneck): 128 -> 64]
     │
     ▼
[Layer Normalization (64) + ReLU]
     │
     ▼
[Output Fully Connected Layer: 64 -> 3] ===> Q-Values: [Thẳng, Trái, Phải]
```

### 4.2. Luận Giải Lựa Chọn Số Nơ-ron
1. **Lớp Fully Connected đầu vào (128 nơ-ron)**:
   - Vector trạng thái có 22 chiều hỗn hợp (nhị phân và liên tục). Ánh xạ 22 -> 128 nơ-ron giúp mạng phóng chiếu đặc trưng vào không gian nhiều chiều hơn, bộc lộ các mối tương quan phi tuyến giữa hướng di chuyển và vị trí mồi.
   - Thêm **Layer Normalization** trước khi đưa vào LSTM giúp triệt tiêu hiện tượng dạt hiệp biến nội tại (Internal Covariate Shift), đưa giá trị về phân phối chuẩn chuẩn hoá giúp các hàm cổng sigmoid và tanh của LSTM không bị bão hoà gradient.
2. **Lớp LSTM (128 units)**:
   - Với bản đồ 14x14 ô (tổng 196 ô), khi rắn đạt độ dài từ 15-40 ô, thân rắn sẽ liên tục quấn gấp khúc tạo ra các "bẫy" hình học. Một mạng nơ-ron tĩnh không thể nhớ được thân rắn đã đi qua đâu trong vài bước trước.
   - Khối tế bào LSTM với **128 chiều ẩn**:
     - **Cell State ($c_t$)**: Lưu giữ vết đường đi và định hướng chiến lược dài hạn nội bộ.
     - **Hidden State ($h_t$)**: Truyền tín hiệu tức thời cho bước kế tiếp và đưa ra lớp phân loại.
     - Số lượng 128 units cân bằng hoàn hảo: đủ dung lượng nhớ ngữ cảnh lịch sử mà số lượng tham số chỉ khoảng ~65.000 trọng số, tính toán cực nhanh ngay trên CPU.
3. **Lớp trung gian (64 nơ-ron)**:
   - Đóng vai trò lớp nút thắt cổ chai (Bottleneck) để chưng cất 128 đặc trưng chuỗi thời gian của LSTM thành 64 đặc trưng quyết định trước khi dự đoán giá trị Q-Value.
4. **Lớp Output (3 nơ-ron)**:
   - Tương ứng trực tiếp với 3 giá trị Q(s, Đi thẳng), Q(s, Rẽ trái), Q(s, Rẽ phải).

---

## 5. Cơ Chế Train Nối Tiếp Giữa Các Lần Train (Continuous / Resumed Training)
Hệ thống tích hợp đầy đủ cơ chế lưu và nạp trạng thái:
- **Thông tin lưu trong Checkpoint**:
  - `policy_net_state` & `target_net_state`: Trọng số của cả 2 mạng nơ-ron.
  - `optimizer_state`: Trạng thái động lượng (Momentum) của thuật toán Adam.
  - `episode` & `total_steps`: Đếm chính xác số tập và số bước để tiếp tục.
  - `epsilon`: Mức độ khám phá hiện tại (đã suy giảm, không bị reset về 1.0 trừ khi người dùng muốn).
  - `best_score`, `score_history`, `loss_history`: Giữ nguyên lịch sử để vẽ biểu đồ liên tục.
- **Tính năng nổi bật**:
  - Tự động lưu checkpoint định kỳ (`latest_checkpoint.pth`) mỗi 50 tập.
  - Tự động lưu bản riêng `best_model.pth` mỗi khi phá kỷ lục điểm số.
  - Bắt tín hiệu **Ctrl+C**: Khi người dùng dừng tiến trình bằng phím tắt, chương trình không bị crash mà lưu ngay lập tức checkpoint của tập hiện tại rồi mới thoát an toàn.
  - Tham số `--resume`: Tiếp tục học ngay từ tập dừng lại lần trước.
  - Tham số `--reset-epsilon`: Cho phép điều chỉnh lại độ khám phá nếu muốn AI tìm kiếm thêm chiến thuật mới khi train nối tiếp.

---

## 6. Hướng Dẫn Sử Dụng

### 6.1. Huấn luyện mới từ đầu (Chế độ chạy ngầm siêu tốc)
```bash
py -3 train.py --episodes 500
```

### 6.2. Huấn luyện nối tiếp từ checkpoint gần nhất
```bash
py -3 train.py --episodes 500 --resume
```

### 6.3. Huấn luyện nối tiếp từ checkpoint cụ thể với epsilon khám phá tối thiểu (0.2%)
```bash
py -3 train.py --episodes 500 --resume --epsilon-min 0.002
```
*(Nếu muốn ép Epsilon nhảy ngay lập tức về 0.002 thay vì giảm dần từ 0.01: thêm `--reset-epsilon 0.002`).*

### 6.4. Huấn luyện có hiển thị giao diện đồ hoạ trực tiếp
```bash
py -3 train.py --episodes 300 --render --fps 40
```

### 6.5. Xem mô hình đã huấn luyện thi đấu trực quan
```bash
py -3 play.py --model checkpoints/best_model.pth --games 10 --fps 15
```
