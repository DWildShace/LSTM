"""
Chương trình xem mô hình AI Rắn săn mồi (DRQN / LSTM) đã huấn luyện thi đấu trực tiếp.
Hiển thị giao diện đồ hoạ Pygame mượt mà, điểm số, và các thông số trạng thái.
"""

import argparse
import os
import sys
import time

# Đảm bảo mã hoá UTF-8 trên Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pygame

from snake_env import SnakeGame
from agent import SnakeAgent


def play():
    parser = argparse.ArgumentParser(description="Xem AI Rắn săn mồi (DRQN/LSTM) chơi game")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth",
                        help="Đường dẫn tới file checkpoint .pth cần nạp (mặc định checkpoints/best_model.pth)")
    parser.add_argument("--games", type=int, default=10, help="Số ván chơi kiểm thử (mặc định 10)")
    parser.add_argument("--fps", type=int, default=15, help="Tốc độ khung hình (mặc định 15 FPS)")
    parser.add_argument("--grid-size", type=int, default=14, help="Kích thước bản đồ (14x14)")
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        # Thử tìm latest_checkpoint nếu best_model chưa có
        alt_path = "checkpoints/latest_checkpoint.pth"
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            print(f"[LỖI] Không tìm thấy file checkpoint tại '{args.model}' hoặc '{alt_path}'.")
            print("Vui lòng chạy 'python train.py' trước để tạo mô hình!")
            return

    env = SnakeGame(grid_size=args.grid_size, cell_size=32, render_mode=True)
    agent = SnakeAgent(
        input_dim=22,
        hidden_fc=128,
        lstm_hidden=128,
        mid_fc=64,
        output_dim=3,
        epsilon=0.0,  # Tắt khám phá, chỉ chọn nước đi tốt nhất
        epsilon_min=0.0,
    )

    print("=" * 60)
    print(f" ĐANG NẠP MÔ HÌNH: {model_path}")
    ep, steps, best_s, _, _ = agent.load_checkpoint(model_path, reset_epsilon=0.0)
    print(f" * Điểm kỷ lục đã lưu trong checkpoint: {best_s}")
    print(f" * Số ván kiểm thử: {args.games} | Tốc độ: {args.fps} FPS")
    print("=" * 60)

    scores = []
    actions_str = ["Đi thẳng", "Rẽ trái", "Rẽ phải"]

    for game_idx in range(1, args.games + 1):
        state = env.reset()
        agent.reset_hidden()
        done = False
        step_count = 0

        while not done:
            step_count += 1
            action = agent.select_action(state, evaluate=True, action_mask=env.get_action_mask())
            next_state, reward, done, score = env.step(action)
            state = next_state

            info_msg = (
                f"Ván: {game_idx}/{args.games} | Nước đi: {actions_str[action]} | "
                f"Kỷ lục checkpoint: {best_s}"
            )
            env.render(fps=args.fps, info_text=info_msg)

        scores.append(env.score)
        print(f"-> Ván {game_idx:2d}: Điểm đạt được = {env.score:2d} (Số bước: {step_count})")
        time.sleep(0.5)

    print("=" * 60)
    print(f" KẾT QUẢ KIỂM THỬ {args.games} VÁN:")
    print(f" * Điểm trung bình: {np.mean(scores):.2f}")
    print(f" * Điểm cao nhất:   {np.max(scores)}")
    print(f" * Điểm thấp nhất:  {np.min(scores)}")
    print("=" * 60)

    pygame.quit()


if __name__ == "__main__":
    play()
