# Trò Chơi Rắn Săn Mồi (Snake Game AI) - A* & LSTM

Dự án nghiên cứu và phát triển AI chơi trò chơi Rắn săn mồi (Snake Game) sử dụng nhiều phương pháp tiếp cận: thuật toán tìm đường cổ điển (A*), Học có giám sát với mạng nơ-ron hồi quy LSTM (Supervised Learning - Imitation Learning), và Học tăng cường sâu với LSTM (Reinforcement Learning).

---

## Cấu Trúc Dự Án

```
tro_choi_ran_san_moi/
├── tim_kiem_A_sao/                   # Thuật toán tìm đường A* chuyên gia (Expert Agent)
├── LSTM_hco_giam_sat/                # Mô hình LSTM học có giám sát từ nhãn của thuật toán A*
├── LSTM_hoc_tang_Cuong/              # Mô hình LSTM học tăng cường (RL) trên bản đồ có vật cản
└── LSTM_hoc_tang_cuong_kocovatcan/   # Mô hình LSTM học tăng cường (RL) trên bản đồ không có vật cản
```

---

## Chi Tiết Các Phân Hệ

### 1. [tim_kiem_A_sao/](./tim_kiem_A_sao)
- Sử dụng thuật toán tìm kiếm đường đi ngắn nhất **A\*** (A-Star) với heuristic Manhattan/Euclidean kết hợp tránh chướng ngại vật và thân rắn.
- Đóng vai trò làm chuyên gia điều hướng và tạo nhãn chuẩn (Ground Truth) phục vụ cho quá trình thu thập tập dữ liệu.

### 2. [LSTM_hco_giam_sat/](./LSTM_hco_giam_sat)
- Huấn luyện mô hình **LSTM 2 tầng** dựa trên phương pháp **Học Bắt Chước (Behavior Cloning)**.
- Dữ liệu huấn luyện: Chuỗi các bước đi tối ưu được trích xuất tự động từ thuật toán A\*.
- Đầu vào: Chuỗi vector trạng thái (vị trí đầu, hướng đi, khoảng cách tới tường, vật cản và mồi).
- Đầu ra: Phân loại hành động (Đi thẳng, Rẽ trái, Rẽ phải).

### 3. [LSTM_hoc_tang_Cuong/](./LSTM_hoc_tang_Cuong)
- Ứng dụng **Học tăng cường sâu (Deep Reinforcement Learning)** kết hợp mạng nhớ dài hạn LSTM để giải quyết bài toán môi trường có vật cản tĩnh và động (thân rắn tự va chạm).

### 4. [LSTM_hoc_tang_cuong_kocovatcan/](./LSTM_hoc_tang_cuong_kocovatcan)
- Huấn luyện AI thông qua học tăng cường trên môi trường lưới không gian mở (không có vật cản tĩnh ngoài biên tường).

---

## Cài Đặt Môi Trường

Yêu cầu Python 3.10+:

```bash
pip install numpy pygame torch tensorflow keras matplotlib
```
