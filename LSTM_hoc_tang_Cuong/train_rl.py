"""
Quy trình Huấn luyện Tự động cho AI Rắn săn mồi bằng Học Tăng Cường (Reinforcement Learning - DRQN)
Sử dụng kiến trúc 2 tầng LSTM với phương pháp Double Q-Learning và Experience Replay.

Hỗ trợ các tùy chọn dòng lệnh:
  --episodes      : Tổng số ván chơi huấn luyện (mặc định 500)
  --batch-size    : Kích thước batch replay (mặc định 64)
  --lr            : Tốc độ học (learning rate, mặc định 0.0005)
  --resume        : Huấn luyện tiếp từ file mô hình đã lưu
  --epsilon-start : Epsilon ban đầu (mặc định 1.0, hoặc 0.2 khi resume)
  --save-interval : Số ván lưu checkpoint một lần (mặc định 25)
"""

import sys
import os
import argparse
import signal
from collections import deque
import numpy as np

# Đảm bảo in tiếng Việt chuẩn trên Windows console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Sử dụng backend torch cho Keras 3
os.environ["KERAS_BACKEND"] = "torch"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import (  # noqa: E402
    TOTAL_EPISODES, BATCH_SIZE, LEARNING_RATE,
    TARGET_UPDATE_GAMES, MAX_STEPS_PER_EPISODE,
    MODEL_PATH
)
from environment import SnakeRLEnvironment  # noqa: E402
from agent import SnakeRLAgent  # noqa: E402


def parse_arguments():
    parser = argparse.ArgumentParser(description="Huấn luyện AI Rắn săn mồi bằng Học Tăng Cường (DRQN)")
    parser.add_argument("--episodes", type=int, default=TOTAL_EPISODES,
                        help=f"Số ván game huấn luyện (mặc định: {TOTAL_EPISODES})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Kích thước mini-batch replay (mặc định: {BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help=f"Tốc độ học Learning Rate (mặc định: {LEARNING_RATE})")
    parser.add_argument("--resume", action="store_true",
                        help="Tải trọng số mô hình trước đó để huấn luyện tiếp")
    parser.add_argument("--epsilon-start", type=float, default=None,
                        help="Xác suất khám phá ban đầu (mặc định: 1.0 hoặc 0.2 khi resume)")
    parser.add_argument("--save-interval", type=int, default=25,
                        help="Số ván lưu checkpoint định kỳ (mặc định: 25)")
    parser.add_argument("--target-update", type=int, default=TARGET_UPDATE_GAMES,
                        help=f"Chu kỳ ván cập nhật Target Network (mặc định: {TARGET_UPDATE_GAMES})")
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 80)
    print("      HUẤN LUYỆN AI RẮN SĂN MỒI - HỌC TĂNG CƯỜNG (DRQN 2 TẦNG LSTM)")
    print("=" * 80)
    print(f"[*] Tổng số ván huấn luyện   : {args.episodes}")
    print(f"[*] Kích thước Mini-batch     : {args.batch_size}")
    print(f"[*] Tốc độ học (Learning Rate): {args.lr}")
    print(f"[*] Chu kỳ cập nhật Target Net: {args.target_update} ván")
    print(f"[*] Nối tiếp huấn luyện (Resume): {args.resume}")
    print("-" * 80)

    # Khởi tạo Môi trường và Agent
    env = SnakeRLEnvironment(num_obstacles=4)
    agent = SnakeRLAgent(
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    # Xử lý Resume mô hình cũ
    if args.resume:
        loaded = agent.load(MODEL_PATH)
        if loaded:
            if args.epsilon_start is not None:
                agent.epsilon = args.epsilon_start
            else:
                agent.epsilon = 0.25  # Giảm khám phá khi đã có kiến thức cơ bản
            print(f"[✓] Đã phục hồi mô hình thành công. Đặt Epsilon ban đầu = {agent.epsilon:.2f}")
    elif args.epsilon_start is not None:
        agent.epsilon = args.epsilon_start

    # Bắt tín hiệu Ctrl+C để tự động lưu mô hình an toàn
    def signal_handler(sig, frame):
        print("\n[!] Nhận tín hiệu dừng từ người dùng (Ctrl+C). Đang lưu mô hình...")
        agent.save(MODEL_PATH)
        print("[✓] Đã lưu mô hình an toàn. Tạm biệt!")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    scores_window = deque(maxlen=20)
    best_score = 0
    total_steps_all = 0

    print(f"{'Ván':^8}|{'Điểm':^6}|{'Kỷ lục':^8}|{'Số bước':^9}|{'Tổng thưởng':^13}|{'Epsilon':^9}|{'Loss TB':^10}|{'ĐTB (20)':^10}")
    print("-" * 80)

    for episode in range(1, args.episodes + 1):
        state = env.reset(randomize_obstacles=True)
        episode_reward = 0.0
        episode_losses = []
        step_count = 0
        done = False

        while not done and step_count < MAX_STEPS_PER_EPISODE:
            # 1. Agent quan sát chuỗi trạng thái và chọn hành động (Epsilon-Greedy)
            action, _ = agent.act(state, evaluate=False)

            # 2. Môi trường thực thi hành động và trả về phần thưởng định hình
            next_state, reward, done, info = env.step(action)

            # 3. Lưu chuyển trạng thái vào Replay Buffer
            agent.remember(state, action, reward, next_state, done)

            # 4. Huấn luyện 1 mini-batch từ bộ nhớ trải nghiệm
            loss = agent.replay()
            if loss > 0:
                episode_losses.append(loss)

            state = next_state
            episode_reward += reward
            step_count += 1
            total_steps_all += 1

        # Cập nhật thông số sau mỗi ván
        score = info["score"]
        scores_window.append(score)
        avg_score = float(np.mean(scores_window))
        avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0

        # Giảm dần epsilon (Exploration Decay)
        agent.decay_epsilon()

        # Định kỳ đồng bộ Target Network
        if episode % args.target_update == 0:
            agent.update_target_network()

        # Kiểm tra phá kỷ lục
        is_new_record = False
        if score > best_score:
            best_score = score
            is_new_record = True
            agent.save(MODEL_PATH)

        # Định kỳ lưu checkpoint
        if episode % args.save_interval == 0 and not is_new_record:
            agent.save(MODEL_PATH)

        record_flag = " ★" if is_new_record else ""
        print(f"{episode:^8}|{score:^6}|{best_score:^6}{record_flag}|{step_count:^9}|{episode_reward:^13.1f}|{agent.epsilon:^9.3f}|{avg_loss:^10.4f}|{avg_score:^10.2f}")

    # Kết thúc toàn bộ quá trình huấn luyện
    print("=" * 80)
    print(f"[✓] HOÀN TẤT HUẤN LUYỆN {args.episodes} VÁN!")
    print(f"[*] Điểm số kỷ lục đạt được: {best_score}")
    print(f"[*] Điểm số trung bình 20 ván cuối: {float(np.mean(scores_window)):.2f}")
    agent.save(MODEL_PATH)
    print("=" * 80)


if __name__ == "__main__":
    main()
