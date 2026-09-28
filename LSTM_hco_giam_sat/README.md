# Huấn Luyện Mô Hình AI LSTM Chơi Rắn Săn Mồi Từ Thuật Toán Tìm Đường A*

Dự án ứng dụng phương pháp **Học Bắt Chước (Imitation Learning / Behavior Cloning)** để huấn luyện một mạng nơ-ron hồi quy sâu **LSTM 2 tầng** tự động chơi trò Rắn săn mồi trên bản đồ $14 \times 14$ có vật cản cố định và thân rắn động, dựa trên dữ liệu nhãn sinh ra từ thuật toán chuyên gia **A\*** trong thư mục [tim_kiem_A_sao](file:///e:/________thuMucHoc/ky_5/tri_tue_nhan_tao/tro_choi_ran_san_moi/tim_kiem_A_sao).

---

## 1. Cơ Chế Tạo Nhãn & Đầu Vào Từ Thuật Toán A* (Expert Labeling)

### 1.1. Bản chất bài toán
Thay vì để AI tự mò mẫm qua hàng triệu lần thử sai ngẫu nhiên (như Reinforcement Learning cơ bản), ta sử dụng **thuật toán tìm đường A\*** trong `tim_kiem_A_sao/astar.py` đóng vai trò là một **Chuyên gia (Expert Agent)**:
* Ở mỗi bước thời gian $t$, trạng thái game $S_t$ được đưa vào thuật toán A\*.
* A\* phân tích vị trí đầu rắn, thân rắn, $3-4$ vật cản cố định và mồi để đưa ra quyết định di chuyển tối ưu $A_t \in \{\text{Đi thẳng}, \text{Rẽ trái}, \text{Rẽ phải}\}$.
* Quyết định $A_t$ của A\* chính là **Nhãn chuẩn (Ground Truth Target Label)** cho trạng thái $S_t$.

```
    Môi trường Game (Bản đồ 14x14)
                 │
                 ▼
       ┌──────────────────┐
       │   Trạng thái St  │
       └─────────┬────────┘
                 │
        ┌────────┴───────────────────────────┐
        ▼                                    ▼
┌───────────────┐                  ┌───────────────────┐
│ Input Feature │                  │  Thuật toán A*    │
│  Extractor    │                  │  (Expert Agent)   │
└───────┬───────┘                  └─────────┬─────────┘
        ▼                                    ▼
  Chuỗi quan sát Xt                  Nhãn hành động Yt
 [T bước liên tiếp]               [Thẳng, Trái, Phải]
        │                                    │
        └─────────────────┬──────────────────┘
                          ▼
            Cặp dữ liệu huấn luyện (Xt, Yt)
```

---

### 1.2. Cấu trúc Nhãn đầu ra (Ground Truth Label $Y$)
A\* xuất ra hành động thông qua hàm `convert_next_pos_to_action` dưới dạng One-Hot vector 3 chiều:
* **Nhãn 0 (Đi thẳng - Straight)**: Vector `[1, 0, 0]`
* **Nhãn 1 (Rẽ trái - Turn Left)**: Vector `[0, 1, 0]` (Góc $90^\circ$ ngược chiều kim đồng hồ)
* **Nhãn 2 (Rẽ phải - Turn Right)**: Vector `[0, 0, 1]` (Góc $90^\circ$ cùng chiều kim đồng hồ)

---

### 1.3. Cấu trúc Đặc trưng đầu vào (Input Features $X_t$)
Tại mỗi bước $t$, module `feature_extractor.py` trích xuất một vector $16$ chiều kết hợp cảm biến va chạm tức thời và thông tin định hướng không gian:

| STT | Đặc trưng | Ý nghĩa | Miền giá trị |
| :---: | :--- | :--- | :---: |
| 1 | `danger_straight` | Có vật cản (tường, vật cản tĩnh, thân rắn) ngay phía trước? | $\{0, 1\}$ |
| 2 | `danger_left` | Có vật cản ngay bên trái? | $\{0, 1\}$ |
| 3 | `danger_right` | Có vật cản ngay bên phải? | $\{0, 1\}$ |
| 4-7 | `dir_up, dir_right, dir_down, dir_left` | Hướng di chuyển hiện tại của đầu rắn (One-hot 4 hướng) | $\{0, 1\}$ |
| 8-11 | `food_up, food_right, food_down, food_left` | Vị trí tương đối của mồi so với đầu rắn (Trên, Phải, Dưới, Trái) | $\{0, 1\}$ |
| 12-13 | `norm_dx, norm_dy` | Khoảng cách tọa độ chuẩn hóa đến mồi: $\frac{x_{food} - x_{head}}{14}, \frac{y_{food} - y_{head}}{14}$ | $[-1.0, 1.0]$ |
| 14-16 | `dist_straight, dist_left, dist_right` | Tầm nhìn an toàn (Raycast): Khoảng cách (số ô) tới vật cản gần nhất chia cho 14 | $[0.0, 1.0]$ |

---

### 1.4. Xử lý Chuỗi thời gian cho Mô hình LSTM (Sequence Sliding Window)
Mô hình LSTM yêu cầu đầu vào dạng chuỗi để học **quán tính di chuyển, vận tốc và sự uốn lượn của thân rắn**.
* Độ dài chuỗi trượt: `SEQ_LEN = 4` bước (có thể điều chỉnh trong `config.py`).
* Tại thời điểm $t$, Input đưa vào mạng là ma trận:
  $$X_{\text{seq}} = \begin{bmatrix} X_{t-3} \\ X_{t-2} \\ X_{t-1} \\ X_t \end{bmatrix} \in \mathbb{R}^{4 \times 16}$$
* Ở những bước đầu game khi $t < 4$, các bước trước được đệm (Zero-padding) bằng vector toàn 0.

---

## 2. Kiến Trúc Mạng Nơ-ron AI LSTM

Mô hình được xây dựng chính xác theo kiến trúc thiết kế:

```
          Input: (batch_size, 4, 16)
                     │
       ┌─────────────▼─────────────┐
       │   LSTM Layer 1 (64 units) │  -> return_sequences=True
       └─────────────┬─────────────┘
       ┌─────────────▼─────────────┐
       │       Dropout (0.2)       │
       └─────────────┬─────────────┘
       ┌─────────────▼─────────────┐
       │   LSTM Layer 2 (32 units) │  -> return_sequences=False
       └─────────────┬─────────────┘
       ┌─────────────▼─────────────┐
       │    Dense Layer (32, ReLU) │
       └─────────────┬─────────────┘
       ┌─────────────▼─────────────┐
       │ Dense Output (3, Softmax) │  -> Xác suất [Thẳng, Trái, Phải]
       └───────────────────────────┘
```

* **LSTM Layer 1 (64 units, `return_sequences=True`)**: Học biểu diễn đặc trưng chuỗi thời gian ở cấp độ cao, xuất ra chuỗi $(4, 64)$.
* **Dropout (0.2)**: Chống hiện tượng quá khớp (Overfitting), ngẫu nhiên triệt tiêu $20\%$ nơ-ron trong quá trình train.
* **LSTM Layer 2 (32 units, `return_sequences=False`)**: Tổng hợp toàn bộ bối cảnh chuỗi thời gian về vector trạng thái ẩn cuối cùng $(32)$.
* **Dense Layer (32 units, ReLU)**: Lớp kết nối đầy đủ phi tuyến tính cô đọng đặc trưng.
* **Dense Output (3 units, Softmax)**: Xuất ra phân phối xác suất hợp lệ ($\sum p_i = 1$) cho 3 toán tử di chuyển. Hành động có xác suất cao nhất $\arg\max(p)$ sẽ được chọn để thực thi.

---

## 3. Cấu Trúc Mã Nguồn Thư Mục `LSTM/`

| Tệp tin | Chức năng |
| :--- | :--- |
| [config.py](file:///e:/________thuMucHoc/ky_5/tri_tue_nhan_tao/tro_choi_ran_san_moi/LSTM/config.py) | Tham số chuỗi `SEQ_LEN=4`, số chiều `FEATURE_DIM=16`, tham số huấn luyện. |
| [feature_extractor.py](file:///e:/________thuMucHoc/ky_5/tri_tue_nhan_tao/tro_choi_ran_san_moi/LSTM/feature_extractor.py) | Trích xuất 16 đặc trưng từ môi trường và cảm biến va chạm. |
| [collect_data.py](file:///e:/________thuMucHoc/ky_5/tri_tue_nhan_tao/tro_choi_ran_san_moi/LSTM/collect_data.py) | Chạy mô phỏng A\* ngầm (Headless) để thu thập mẫu và tạo nhãn cực nhanh (~5.000 mẫu/giây). |
| [model.py](file:///e:/________thuMucHoc/ky_5/tri_tue_nhan_tao/tro_choi_ran_san_moi/LSTM/model.py) | Định nghĩa kiến trúc Keras/Torch 2 tầng LSTM theo đúng sơ đồ. |
| [train.py](file:///e:/________thuMucHoc/ky_5/tri_tue_nhan_tao/tro_choi_ran_san_moi/LSTM/train.py) | Huấn luyện mô hình, cân bằng trọng số lớp (Class Weights), lưu mô hình `.keras`. |
| [play_ai.py](file:///e:/________thuMucHoc/ky_5/tri_tue_nhan_tao/tro_choi_ran_san_moi/LSTM/play_ai.py) | Giao diện đồ họa Pygame xem AI LSTM tự chơi với Dashboard hiển thị thanh xác suất Softmax thời gian thực. |
| [run_collect.bat](file:///e:/________thuMucHoc/ky_5/tri_tue_nhan_tao/tro_choi_ran_san_moi/LSTM/run_collect.bat) | File 1-click thu thập dữ liệu nhãn từ A\*. |
| [run_train.bat](file:///e:/________thuMucHoc/ky_5/tri_tue_nhan_tao/tro_choi_ran_san_moi/LSTM/run_train.bat) | File 1-click chạy huấn luyện mô hình. |
| [run_play.bat](file:///e:/________thuMucHoc/ky_5/tri_tue_nhan_tao/tro_choi_ran_san_moi/LSTM/run_play.bat) | File 1-click mở game quan sát AI thi đấu. |
| [run_all.bat](file:///e:/________thuMucHoc/ky_5/tri_tue_nhan_tao/tro_choi_ran_san_moi/LSTM/run_all.bat) | File 1-click chạy toàn bộ quy trình: Thu thập -> Train -> Chạy game. |

---

## 4. Hướng Dẫn Sử Dụng

### Cách 1: Chạy tự động toàn bộ quy trình (1-Click)
Nhấp đúp chuột vào file [run_all.bat](file:///e:/________thuMucHoc/ky_5/tri_tue_nhan_tao/tro_choi_ran_san_moi/LSTM/run_all.bat).

### Cách 2: Chạy từng bước qua Terminal
```powershell
# Bước 1: Di chuyển vào thư mục LSTM
cd e:\________thuMucHoc\ky_5\tri_tue_nhan_tao\tro_choi_ran_san_moi\LSTM

# Bước 2: Thu thập dữ liệu và gán nhãn tự động từ thuật toán A* (150 - 250 ván)
py -3 collect_data.py

# Bước 3: Huấn luyện mô hình AI LSTM
py -3 train.py

# Bước 4: Mở game với giao diện Pygame để xem AI LSTM tự chơi
py -3 play_ai.py
```
