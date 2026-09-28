"""
Tạo nhãn và thu thập tập dữ liệu (Dataset Generation) từ thuật toán tìm kiếm A*.
Phương pháp: Huấn luyện bắt chước (Imitation Learning / Behavior Cloning).
- Input: Chuỗi T bước quan sát liên tiếp [X_{t-T+1}, ..., X_t]
- Label (Nhãn): Hành động tương ứng do A* lựa chọn [Thẳng, Trái, Phải]
"""

import sys
import os
import time
from collections import deque
import numpy as np

# Đảm bảo in tiếng Việt không bị lỗi charmap trên Windows console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Thiết lập đường dẫn thư mục để import an toàn
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
ASTAR_DIR = os.path.join(PARENT_DIR, "tim_kiem_A_sao")

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if ASTAR_DIR not in sys.path:
    sys.path.insert(0, ASTAR_DIR)

from snake_game import SnakeGame  # noqa: E402
from astar import default_finder  # noqa: E402
from config import (  # noqa: E402
    SEQ_LEN, FEATURE_DIM, NUM_COLLECT_GAMES,
    MAX_STEPS_PER_GAME, DATASET_PATH
)
from feature_extractor import extract_features  # noqa: E402


import argparse  # noqa: E402


def collect_expert_demonstrations(num_games: int = NUM_COLLECT_GAMES,
                                   seq_len: int = SEQ_LEN,
                                   max_steps: int = MAX_STEPS_PER_GAME,
                                   append_existing: bool = False):
    """
    Chạy thuật toán A* trong chế độ chạy ngầm (Headless - không mở GUI Pygame)
    để thu thập hàng chục ngàn / hàng triệu cặp (Input Sequence, Label Action).
    """
    print("=" * 70)
    print(" BẮT ĐẦU THU THẬP DỮ LIỆU & TẠO NHÃN TỪ THUẬT TOÁN A* (EXPERT)")
    print(f" - Số ván chơi mô phỏng: {num_games}")
    print(f" - Độ dài chuỗi LSTM (seq_len): {seq_len} bước")
    print(f" - Kích thước đặc trưng mỗi bước (feature_dim): {FEATURE_DIM}")
    print(f" - Chế độ ghi: {'GỘP NỐI TIẾP (APPEND)' if append_existing else 'GHI MỚI TOÀN BỘ (OVERWRITE)'}")
    print("=" * 70)

    game = SnakeGame(num_obstacles=4)
    all_sequences = []
    all_labels = []

    # Thống kê phân bố nhãn
    action_counts = {0: 0, 1: 0, 2: 0}  # 0: Thẳng, 1: Trái, 2: Phải
    total_scores = []
    start_time = time.time()

    for game_idx in range(1, num_games + 1):
        game.reset(randomize_obstacles=True)
        step_count = 0

        # Khởi tạo buffer trượt (Sliding Window Queue) chứa seq_len bước
        history_buffer = deque(maxlen=seq_len)
        for _ in range(seq_len):
            history_buffer.append(np.zeros(FEATURE_DIM, dtype=np.float32))

        while not game.is_game_over and step_count < max_steps:
            step_count += 1

            # 1. Trích xuất vector đặc trưng trạng thái tại bước t
            current_feat = extract_features(
                head=game.head,
                current_dir=game.direction,
                snake_body=game.body,
                obstacles=game.obstacles,
                food=game.food
            )

            # 2. Đẩy vào sliding window buffer
            history_buffer.append(current_feat)
            seq_input = np.array(history_buffer, dtype=np.float32)

            # 3. LẤY NHÃN TỪ A* (EXPERT LABELING):
            expert_action, _, _, _ = default_finder.get_next_action(
                head=game.head,
                current_dir=game.direction,
                food=game.food,
                obstacles=game.obstacles,
                snake_body=game.body
            )
            action_idx = int(np.argmax(expert_action))
            action_counts[action_idx] += 1

            # 4. Lưu cặp (Input Sequence, Label) vào tập dữ liệu
            all_sequences.append(seq_input)
            all_labels.append(expert_action)

            # 5. Cho rắn thực hiện bước đi trong game
            _, is_over, _ = game.play_step(expert_action)

        total_scores.append(game.score)

        # In tiến độ định kỳ
        log_interval = 25 if num_games <= 200 else (50 if num_games <= 1000 else 100)
        if game_idx % log_interval == 0 or game_idx == num_games:
            elapsed = time.time() - start_time
            recent_scores = total_scores[-log_interval:]
            print(f"[*] Đã thu thập: {game_idx:4d}/{num_games} ván | "
                  f"Mẫu đợt này: {len(all_sequences):7d} | "
                  f"Điểm TB: {np.mean(recent_scores):.1f} (Cao nhất: {max(recent_scores)}) | "
                  f"Thời gian: {elapsed:.1f}s")

    # Chuyển đổi sang mảng numpy tối ưu
    new_X = np.array(all_sequences, dtype=np.float32)  # Shape: (N, seq_len, feature_dim)
    new_Y = np.array(all_labels, dtype=np.float32)     # Shape: (N, 3) (One-hot)

    # Xử lý chế độ Append nếu người dùng yêu cầu
    if append_existing and os.path.exists(DATASET_PATH):
        try:
            print(f"\n[*] Đang gộp nối tiếp vào tập dữ liệu cũ tại: {DATASET_PATH}")
            old_data = np.load(DATASET_PATH)
            old_X = old_data['X']
            old_Y = old_data['Y']
            X = np.concatenate([old_X, new_X], axis=0)
            Y = np.concatenate([old_Y, new_Y], axis=0)
            print(f" - Dữ liệu cũ: {len(old_X)} mẫu + Dữ liệu mới: {len(new_X)} mẫu = {len(X)} mẫu")
        except Exception as e:
            print(f"[!] Không thể gộp dữ liệu cũ ({e}), lưu dữ liệu mới!")
            X, Y = new_X, new_Y
    else:
        X, Y = new_X, new_Y

    Y_cls = np.argmax(Y, axis=1)
    total_samples = len(X)
    cls_counts = np.bincount(Y_cls, minlength=3)

    print("\n" + "=" * 70)
    print(" TỔNG KẾT TẬP DỮ LIỆU HUẤN LUYỆN:")
    print(f" - Tổng số mẫu tích lũy (Total Samples): {total_samples}")
    print(f" - Input Shape  (X): {X.shape} -> (Batch, Seq_Len={seq_len}, Dim={FEATURE_DIM})")
    print(f" - Target Shape (Y): {Y.shape} -> (Batch, Actions=3)")
    print(f" - Nhãn 0 (Đi thẳng): {cls_counts[0]:7d} ({cls_counts[0]/total_samples*100:5.1f}%)")
    print(f" - Nhãn 1 (Rẽ trái) : {cls_counts[1]:7d} ({cls_counts[1]/total_samples*100:5.1f}%)")
    print(f" - Nhãn 2 (Rẽ phải) : {cls_counts[2]:7d} ({cls_counts[2]/total_samples*100:5.1f}%)")
    print(f" - Điểm số A* cao nhất đợt này: {max(total_scores)}")
    print("=" * 70)

    # Lưu tập dữ liệu ra file nén .npz
    np.savez_compressed(DATASET_PATH, X=X, Y=Y, Y_cls=Y_cls)
    print(f"[✓] Đã lưu thành công tập dữ liệu vào: {DATASET_PATH}\n")
    return X, Y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thu thập dữ liệu và tạo nhãn từ thuật toán A*")
    parser.add_argument("-g", "--games", type=int, default=NUM_COLLECT_GAMES,
                        help=f"Số ván chơi A* để thu thập (Mặc định: {NUM_COLLECT_GAMES})")
    parser.add_argument("-a", "--append", action="store_true",
                        help="Gộp thêm vào tập dữ liệu cũ thay vì ghi đè")
    parser.add_argument("--steps", type=int, default=MAX_STEPS_PER_GAME,
                        help=f"Số bước tối đa mỗi ván (Mặc định: {MAX_STEPS_PER_GAME})")
    args = parser.parse_args()

    collect_expert_demonstrations(
        num_games=args.games,
        append_existing=args.append,
        max_steps=args.steps
    )
