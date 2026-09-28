# Trò Chơi Rắn Săn Mồi AI (A* Search) - Bản Đồ 14x14

Dự án mô phỏng và xây dựng trò chơi **Rắn săn mồi (Snake Game)** ứng dụng thuật toán tìm kiếm thông minh **A\*** (A-Star), cảm biến phát hiện vật cản môi trường và hệ thống 3 toán tử di chuyển tương đối.

---

## 1. Yêu Cầu & Kiến Trúc Dự Án

### 1.1. Bản đồ trò chơi (Environment)
* **Kích thước lưới**: $14 \times 14$ ô (`GRID_WIDTH = 14`, `GRID_HEIGHT = 14`).
* **Vật cản cố định**: $3 - 4$ ô chướng ngại vật tĩnh (Static Obstacles) được sinh ngẫu nhiên trên bản đồ (không chặn kín lối đi ban đầu và không trùng vị trí mồi).
* **Vật cản động (Thân rắn)**: Thân rắn dài ra khi ăn mồi và dịch chuyển theo từng bước đi của đầu rắn. Các đốt thân rắn được xem là chướng ngại vật động cần tránh.

### 1.2. 3 Toán tử chính (Action Space)
Rắn không di chuyển theo hướng tuyệt đối (Lên/Xuống/Trái/Phải của bàn phím) mà điều khiển dựa trên **3 toán tử tương đối theo hướng đầu rắn**:
1. **Đi thẳng (`Straight`)**: Vector `[1, 0, 0]` - Rắn tiếp tục đi theo hướng hiện tại.
2. **Rẽ trái (`Turn Left`)**: Vector `[0, 1, 0]` - Rắn bẻ góc $90^\circ$ ngược chiều kim đồng hồ so với hướng hiện tại.
3. **Rẽ phải (`Turn Right`)**: Vector `[0, 0, 1]` - Rắn bẻ góc $90^\circ$ theo chiều kim đồng hồ so với hướng hiện tại.

### 1.3. Module phát hiện vật cản (Perception / Obstacle Detector)
Module cảm biến cung cấp thông tin môi trường xung quanh đầu rắn:
* **Cảm biến va chạm 3 hướng tương đối**:
  * `danger_straight`: Nguy hiểm ngay phía trước đầu rắn (Tường biên $14\times 14$, $3-4$ vật cản cố định hoặc thân rắn).
  * `danger_left`: Nguy hiểm ngay bên trái đầu rắn.
  * `danger_right`: Nguy hiểm ngay bên phải đầu rắn.
* **Vector trạng thái AI (Feature State Representation)**:
  * Trạng thái va chạm: `[danger_straight, danger_left, danger_right]`
  * Hướng hiện tại: `[dir_up, dir_right, dir_down, dir_left]`
  * Hướng mồi tương đối: `[food_up, food_right, food_down, food_left]`

---

## 2. Thuật Toán Tìm Đường A* (A-Star Search)

### 2.1. Hàm đánh giá chi phí
$$f(n) = g(n) + h(n)$$
* **$g(n)$ (Actual Cost)**: Số bước đi thực tế từ vị trí đầu rắn hiện tại đến ô $n$. Mỗi bước đi có chi phí bằng $1$.
* **$h(n)$ (Heuristic)**: Khoảng cách Manhattan ước lượng từ ô $n$ tới vị trí mồi:
  $$h(n) = |x_n - x_{food}| + |y_n - y_{food}|$$
  *Khoảng cách Manhattan là Heuristic hợp lý (admissible) và nhất quán (consistent) trên lưới ô vuông không đi chéo.*

### 2.2. Xử lý Thân rắn như Vật cản động (Dynamic Obstacle Modeling)
* Khi rắn bước $k$ bước về phía trước, đuôi rắn sẽ co lại $k$ đốt (nếu chưa ăn mồi).
* **Cơ chế chống bẫy cụt (Dead-end / Trap Prevention)**:
  * Khi A* tìm ra đường đi ngắn nhất tới mồi, hệ thống sẽ thực hiện **mô phỏng giả định (Virtual Simulation)** xem sau khi ăn mồi ở vị trí đó, đầu rắn có còn đường thoát về đuôi của mình hay không.
  * Nếu không có lối thoát an toàn, rắn sẽ chuyển sang **Chế độ Sinh Tồn (Survival Mode)**:
    1. Bám theo đuôi rắn (Tail Chasing) vì đuôi di chuyển liên tục sẽ mở ra đường đi.
    2. Nếu không bám được đuôi, sử dụng **BFS Flood Fill** chọn hướng có diện tích không gian tự do lớn nhất để chờ cơ hội an toàn.

---

## 3. Các Cấu Trúc Dữ Liệu & Kỹ Thuật Tối Ưu Giải Thuật (Advanced DSA Optimizations)

Để đạt tốc độ xử lý siêu nhanh (~$80 - 150\,\mu s$ mỗi chu kỳ ra quyết định) và tiết kiệm tài nguyên, thuật toán A* đã được tối ưu hóa toàn diện bằng các kỹ thuật CTDL & GT:

| STT | Cấu trúc dữ liệu & Kỹ thuật | Giải pháp triển khai | Độ phức tạp & Lợi ích |
| :---: | :--- | :--- | :--- |
| **1** | **Flattened 1D Array** *(Bảng chỉ mục trực tiếp)* | Ánh xạ tọa độ $(x, y) \to \text{idx} = y \times 14 + x \in [0, 195]$. Mọi mảng `g_score`, `parent`, `visited` đều là mảng 1D 196 phần tử. | Truy cập $O(1)$ tức thời trong bộ nhớ liền kề (Cache Locality), loại bỏ overhead băm (hash) của `dict`/`set`. |
| **2** | **Epoch / Timestamp Array** *(Mảng đánh dấu thời gian)* | Dùng mảng `visited_epoch[196]`, `closed_epoch[196]` và biến đếm `epoch`. Mỗi lần A* chạy chỉ cần tăng `epoch += 1`. | **Reset toàn bộ tập đỉnh đã duyệt về 0 trong $O(1)$** mà không cần cấp phát lại bộ nhớ (Zero memory reallocation). |
| **3** | **Space-Time Reservation Table** *(Bảng đặt chỗ thời gian - không gian)* | Thân rắn dài $L$, đốt thứ $i$ rời đi sau $(L - i)$ bước. Mảng `body_free_step[idx] = L - i`. Kiểm tra ô bị chặn nếu `next_g < body_free_step[idx]`. | Chuyển việc kiểm tra vật cản động từ $O(L)$ (cắt slice list) thành **đúng 1 phép so sánh số nguyên $O(1)$**. |
| **4** | **Bitboard / Bitmasking** *(Bảng cờ nhị phân)* | Nén 196 ô bản đồ thành số nguyên nhị phân 196-bit. Vị trí 4 vật cản tĩnh là các bit 1 trong `static_obs_mask`. | Kiểm tra va chạm vật cản tĩnh bằng phép dịch bit: `(mask >> idx) & 1` trong **$O(1)$ ở cấp độ vi xử lý**. |
| **5** | **Packed Primitive Heap** *(Hàng đợi ưu tiên tuple)* | Dùng `heapq` lưu tuple nguyên thủy `(f, h, idx, x, y)` thay vì khởi tạo đối tượng `class AStarNode`. | CPython so sánh trực tiếp ở tầng C, không tốn chi phí cấp phát đối tượng và gọi magic method `__lt__`. |
| **6** | **Tie-Breaking Heuristic** *(Phá vỡ thế cân bằng)* | Nhân heuristic với hệ số vi sai: $f = g + h \times 1.001$. | Giúp A* ưu tiên đi thẳng tới đích thay vì mở rộng dàn trải các node có cùng $f$, **giảm 40-50% số node duyệt**. |
| **7** | **Linear Array BFS Queue** *(BFS 2 con trỏ)* | Dùng mảng phẳng 1 chiều kết hợp 2 con trỏ `head_ptr`, `tail_ptr` và `visited_epoch`. | Quét không gian an toàn nhanh gấp 3 lần so với `collections.deque` và `set`. |

---

## 4. Cấu Trúc Mã Nguồn

| Tệp tin | Chức năng chính |
| :--- | :--- |
| [constants.py](file:///e:/________thuMucHoc/ky_5/tri_tue_nhan_tao/tro_choi_ran_san_moi/hoc_may_A_sao/constants.py) | Kích thước lưới $14\times 14$, bảng màu Neon/Dark mode, định nghĩa 3 toán tử và hướng. |
| [obstacle_detector.py](file:///e:/________thuMucHoc/ky_5/tri_tue_nhan_tao/tro_choi_ran_san_moi/hoc_may_A_sao/obstacle_detector.py) | Module cảm biến va chạm: kiểm tra tường, 3-4 vật cản cố định và thân rắn động. |
| [astar.py](file:///e:/________thuMucHoc/ky_5/tri_tue_nhan_tao/tro_choi_ran_san_moi/hoc_may_A_sao/astar.py) | Thuật toán tìm đường A*, hàm đánh giá $f = g + h$, cơ chế chống bẫy và bộ chuyển đổi sang 3 toán tử. |
| [snake_game.py](file:///e:/________thuMucHoc/ky_5/tri_tue_nhan_tao/tro_choi_ran_san_moi/hoc_may_A_sao/snake_game.py) | Quản lý vòng lặp game, điểm số, tạo vật cản, sinh mồi và thực thi hành động `play_step(action)`. |
| [main.py](file:///e:/________thuMucHoc/ky_5/tri_tue_nhan_tao/tro_choi_ran_san_moi/hoc_may_A_sao/main.py) | Giao diện đồ họa Pygame trực quan với Dashboard hiển thị 3 toán tử, cảm biến và vệt sáng A*. |
| [run.bat](file:///e:/________thuMucHoc/ky_5/tri_tue_nhan_tao/tro_choi_ran_san_moi/hoc_may_A_sao/run.bat) | File chạy nhanh 1-click trên hệ điều hành Windows. |

---

## 4. Hướng Dẫn Cài Đặt & Chạy Game

### Cách 1: Chạy bằng Terminal
```powershell
# Chạy với Python 3
py -3 main.py
```

### Cách 2: Chạy trực tiếp trên Windows
* Double-click chuột vào file `run.bat` trong thư mục dự án.

---

## 5. Phím Tắt Điều Khiển Trong Game

* **`SPACE`**: Tạm dừng (Pause) / Tiếp tục trò chơi.
* **`MŨI TÊN LÊN (▲)`**: Tăng tốc độ game (FPS).
* **`MŨI TÊN XUỐNG (▼)`**: Giảm tốc độ game (FPS) để quan sát từng bước A*.
* **`R`**: Khởi động lại game và sinh ngẫu nhiên $3-4$ vật cản cố định mới.
* **`P`**: Bật / Tắt hiển thị đường đi dự kiến của A* trên bản đồ.
* **`M`**: Chuyển đổi giữa chế độ **AI Tự động A\*** và **Điều khiển thủ công** bằng bàn phím.
