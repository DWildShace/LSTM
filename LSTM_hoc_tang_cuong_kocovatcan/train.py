"""
Chương trình huấn luyện mô hình Học tăng cường DRQN (LSTM) cho trò chơi Rắn săn mồi.
Hỗ trợ:
1. Huấn luyện nối tiếp (Continuous / Resumed Training): Tự động nạp hoặc chỉ định checkpoint để tiếp tục học.
2. Lưu checkpoint an toàn khi bấm Ctrl+C hoặc định kỳ.
3. Chế độ giao diện trực quan (Pygame) hoặc chế độ chạy ngầm siêu tốc (Headless).
4. Vẽ đồ thị tiến trình học tập (Score, Average Score, Loss) lưu ra file ảnh.
"""

import argparse
import os
import signal
import sys

# Đảm bảo mã hoá UTF-8 trên Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Vẽ đồ thị không cần màn hình hiển thị
import matplotlib.pyplot as plt

from snake_env import SnakeGame
from agent import SnakeAgent


def plot_metrics(score_history, loss_history, output_path="training_curve.png"):
    """Vẽ và cập nhật biểu đồ tiến trình huấn luyện"""
    if not score_history:
        return

    plt.figure(figsize=(12, 5))

    # Đồ thị điểm số
    plt.subplot(1, 2, 1)
    plt.title("Điểm số qua từng Episode (Score & Mean Score)")
    plt.plot(score_history, label="Score", alpha=0.35, color="gray")

    # Đường trung bình động 50 tập
    window = min(50, len(score_history))
    if len(score_history) >= window:
        means = [
            np.mean(score_history[max(0, i - window) : i + 1])
            for i in range(len(score_history))
        ]
        plt.plot(means, label=f"Trung bình {window} tập", color="royalblue", linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Điểm (Số mồi ăn được)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # Đồ thị suy giảm hàm mất mát (Loss)
    plt.subplot(1, 2, 2)
    plt.title("Hàm mất mát huấn luyện (Huber Loss)")
    if loss_history:
        # Lấy mẫu trung bình để đường vẽ mượt mà
        step = max(1, len(loss_history) // 500)
        plt.plot(loss_history[::step], color="crimson", alpha=0.8, linewidth=1.5)
        plt.xlabel(f"Bước tối ưu (x{step})")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def train():
    parser = argparse.ArgumentParser(description="Huấn luyện AI Rắn săn mồi với LSTM & Học tăng cường")
    parser.add_argument("--episodes", type=int, default=600, help="Tổng số episode huấn luyện (mặc định 600)")
    parser.add_argument("--grid-size", type=int, default=14, help="Kích thước bản đồ (mặc định 14x14)")
    parser.add_argument("--render", action="store_true", help="Bật hiển thị đồ hoạ Pygame khi train")
    parser.add_argument("--render-every", type=int, default=1, help="Chỉ render đồ hoạ mỗi N tập")
    parser.add_argument("--fps", type=int, default=40, help="Tốc độ khung hình khi render")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Thư mục lưu trữ checkpoint")
    parser.add_argument("--resume", type=str, nargs="?", const="latest", default=None,
                        help="Nối tiếp quá trình train: 'latest' hoặc đường dẫn tới file .pth")
    parser.add_argument("--reset-epsilon", type=float, default=None,
                        help="Thiết lập lại giá trị epsilon khi nối tiếp (vd: 0.2 để khám phá thêm)")
    parser.add_argument("--save-every", type=int, default=50, help="Lưu checkpoint mỗi N tập")
    parser.add_argument("--seq-len", type=int, default=8, help="Độ dài chuỗi thời gian cho LSTM (mặc định 8)")
    parser.add_argument("--batch-size", type=int, default=32, help="Kích thước batch kinh nghiệm")
    parser.add_argument("--train-freq", type=int, default=4,
                        help="Tần suất cập nhật trọng số LSTM (mỗi N bước môi trường, mặc định 4 - chuẩn DeepMind)")
    parser.add_argument("--epsilon-min", type=float, default=0.002,
                        help="Mức khám phá tối thiểu (mặc định 0.002 = 0.2%%)")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    latest_checkpoint_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pth")
    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    plot_path = os.path.join(args.checkpoint_dir, "training_curve.png")

    # Khởi tạo môi trường trò chơi
    env = SnakeGame(grid_size=args.grid_size, cell_size=28, render_mode=args.render)

    # Khởi tạo tác nhân DRQN
    agent = SnakeAgent(
        input_dim=22,
        hidden_fc=128,
        lstm_hidden=128,
        mid_fc=64,
        output_dim=3,
        lr=0.0005,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=args.epsilon_min,
        epsilon_decay=0.996,
        buffer_capacity=3000,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )

    start_episode = 1
    total_steps = 0
    best_score = 0
    score_history = []
    loss_history = []

    # Cơ chế Train nối tiếp (Resume)
    if args.resume is not None:
        target_ckpt = latest_checkpoint_path if args.resume == "latest" else args.resume
        if os.path.exists(target_ckpt):
            ckpt_ep, ckpt_steps, ckpt_best, ckpt_scores, ckpt_losses = agent.load_checkpoint(
                target_ckpt, reset_epsilon=args.reset_epsilon
            )
            start_episode = ckpt_ep + 1
            total_steps = ckpt_steps
            best_score = ckpt_best
            score_history = ckpt_scores
            loss_history = ckpt_losses
        else:
            print(f"[CẢNH BÁO] Không tìm thấy checkpoint tại {target_ckpt}. Bắt đầu train mới từ đầu!")

    # Thiết lập cơ chế bắt sự kiện dừng an toàn Ctrl+C
    interrupted = False

    def handle_interrupt(sig, frame):
        nonlocal interrupted
        print("\n[THÔNG BÁO] Đã nhận tín hiệu ngắt (Ctrl+C). Đang tiến hành lưu checkpoint trước khi thoát...")
        interrupted = True

    signal.signal(signal.SIGINT, handle_interrupt)

    print("=" * 70)
    print(f" BẮT ĐẦU HUẤN LUYỆN AI RẮN SĂN MỒI (DRQN / LSTM) TRÊN BẢN ĐỒ {args.grid_size}x{args.grid_size}")
    print(f" * Thiết bị tính toán: {agent.device}")
    print(f" * Tập bắt đầu: {start_episode} -> Kết thúc: {start_episode + args.episodes - 1}")
    print(f" * Epsilon hiện tại: {agent.epsilon:.4f} (Epsilon tối thiểu: {agent.epsilon_min:.4f} = {agent.epsilon_min*100:.1f}%)")
    print(f" * Chế độ giao diện Pygame: {'BẬT' if args.render else 'TẮT (Chạy ngầm siêu tốc)'}")
    print(f" * File checkpoint nối tiếp: {latest_checkpoint_path}")
    print("=" * 70)

    end_episode = start_episode + args.episodes

    for episode in range(start_episode, end_episode):
        if interrupted:
            break

        state = env.reset()
        agent.reset_hidden()  # Đặt lại bộ nhớ ẩn (h_0, c_0) của LSTM cho ván mới
        done = False
        episode_transitions = []
        episode_reward = 0.0

        should_render = args.render and (episode % args.render_every == 0)
        if should_render:
            env.enable_render()

        while not done and not interrupted:
            total_steps += 1

            # Lựa chọn hành động (Epsilon-Greedy kết hợp Action Masking)
            action = agent.select_action(state, action_mask=env.get_action_mask())

            # Thực hiện bước đi trong môi trường
            next_state, reward, done, score = env.step(action)
            episode_reward += reward

            # Lưu vào danh sách các bước của episode hiện tại
            episode_transitions.append((state, action, reward, next_state, done))

            # Thực hiện tối ưu hoá mạng nơ-ron mỗi train_freq bước (chuẩn DQN/DRQN)
            if total_steps % args.train_freq == 0:
                loss = agent.train_step()
                if loss is not None:
                    loss_history.append(loss)

            state = next_state

            # Hiển thị giao diện nếu bật
            if should_render:
                avg_last = np.mean(score_history[-20:]) if score_history else 0.0
                info_msg = f"Ep: {episode} | Eps: {agent.epsilon:.3f} | Kỷ lục: {best_score} | TB: {avg_last:.1f}"
                env.render(fps=args.fps, info_text=info_msg)

        # Lưu toàn bộ episode vào Sequential Replay Buffer
        agent.memory.push(episode_transitions)

        # Giảm dần độ khám phá epsilon
        agent.decay_epsilon()

        # Cập nhật thống kê
        score_history.append(env.score)
        if env.score > best_score:
            best_score = env.score
            # Lưu model kỷ lục riêng
            agent.save_checkpoint(
                best_model_path,
                episode=episode,
                total_steps=total_steps,
                best_score=best_score,
                score_history=score_history,
                loss_history=loss_history,
            )
            print(f">>> [KỶ LỤC MỚI!] Episode {episode}: Đạt điểm {best_score}! Đã lưu vào {best_model_path}")

        # In thông tin tiến trình huấn luyện
        if episode % 10 == 0 or episode == start_episode:
            mean_score_50 = np.mean(score_history[-50:])
            current_loss = np.mean(loss_history[-100:]) if loss_history else 0.0
            print(
                f"Ep {episode:4d}/{end_episode - 1} | Điểm: {env.score:2d} | "
                f"TB 50 tập: {mean_score_50:5.2f} | Kỷ lục: {best_score:2d} | "
                f"Epsilon: {agent.epsilon:.4f} | Loss: {current_loss:.4f} | Bước: {total_steps}"
            )

        # Lưu checkpoint nối tiếp định kỳ
        if episode % args.save_every == 0:
            agent.save_checkpoint(
                latest_checkpoint_path,
                episode=episode,
                total_steps=total_steps,
                best_score=best_score,
                score_history=score_history,
                loss_history=loss_history,
            )
            plot_metrics(score_history, loss_history, plot_path)

    # Lưu checkpoint kết thúc phiên train
    print("\n" + "=" * 70)
    print(" LƯU CHECKPOINT CUỐI CÙNG & HOÀN TẤT PHIÊN HUẤN LUYỆN")
    final_episode = episode if not interrupted else episode - 1
    agent.save_checkpoint(
        latest_checkpoint_path,
        episode=final_episode,
        total_steps=total_steps,
        best_score=best_score,
        score_history=score_history,
        loss_history=loss_history,
    )
    plot_metrics(score_history, loss_history, plot_path)
    print(f" * Checkpoint mới nhất: {latest_checkpoint_path}")
    print(f" * Checkpoint điểm cao nhất: {best_model_path}")
    print(f" * Đồ thị huấn luyện: {plot_path}")
    print(f" * Tập đã train: {final_episode} | Điểm kỷ lục: {best_score}")
    print("=" * 70)


if __name__ == "__main__":
    train()
