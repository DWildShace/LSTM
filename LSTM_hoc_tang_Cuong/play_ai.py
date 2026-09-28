"""
Chương trình chơi Rắn săn mồi bằng Mô hình Trí tuệ Nhân tạo Học Tăng Cường (DRQN).
Mô hình ước lượng hàm giá trị kỳ vọng Q(s, a) cho 3 toán tử:
  0: Đi thẳng [1, 0, 0]
  1: Rẽ trái  [0, 1, 0]
  2: Rẽ phải  [0, 0, 1]
AI chọn hành động có Q-value cao nhất: a* = argmax_a Q(s, a).
Giao diện Pygame hiển thị thời gian thực các giá trị Q-value và thanh trực quan.
"""

import sys
import os
from collections import deque
import numpy as np
import pygame

# Đảm bảo in tiếng Việt chuẩn trên Windows console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Sử dụng backend torch cho Keras 3
os.environ["KERAS_BACKEND"] = "torch"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
ASTAR_DIR = os.path.join(PARENT_DIR, "tim_kiem_A_sao")

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if ASTAR_DIR not in sys.path:
    sys.path.insert(0, ASTAR_DIR)

from constants import (  # noqa: E402
    GRID_WIDTH, GRID_HEIGHT, CELL_SIZE,
    BOARD_WIDTH, BOARD_HEIGHT, SIDEBAR_WIDTH, WINDOW_WIDTH, WINDOW_HEIGHT,
    DIR_UP, DIR_RIGHT, DIR_LEFT,
    COLOR_BG_DARK, COLOR_BOARD_BG, COLOR_GRID_LINE, COLOR_SIDEBAR_BG, COLOR_PANEL_BG,
    COLOR_SNAKE_HEAD, COLOR_SNAKE_BODY_START, COLOR_SNAKE_BODY_END,
    COLOR_SNAKE_EYES, COLOR_SNAKE_PUPIL,
    COLOR_FOOD, COLOR_FOOD_GLOW,
    COLOR_OBSTACLE, COLOR_OBSTACLE_BORDER, COLOR_OBSTACLE_STRIPE,
    COLOR_TEXT_WHITE, COLOR_TEXT_MUTED,
    COLOR_DANGER, COLOR_SAFE,
    ACTION_STRAIGHT, ACTION_LEFT, ACTION_RIGHT,
    DEFAULT_FPS
)
from obstacle_detector import ObstacleDetector  # noqa: E402
from snake_game import SnakeGame  # noqa: E402
from config import SEQ_LEN, FEATURE_DIM, MODEL_PATH, MODEL_H5_PATH  # noqa: E402
from feature_extractor import extract_features  # noqa: E402


class SnakeAIDRQNVisualizer:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("AI Rắn Săn Mồi - Học Tăng Cường DRQN (2 Tầng LSTM) | 14x14")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        # Font chữ hỗ trợ tiếng Việt trên Windows
        self.font_title = pygame.font.SysFont('segoeui', 20, bold=True)
        self.font_sub = pygame.font.SysFont('segoeui', 14, bold=True)
        self.font_regular = pygame.font.SysFont('segoeui', 13)
        self.font_small = pygame.font.SysFont('segoeui', 11)
        self.font_bold = pygame.font.SysFont('segoeui', 13, bold=True)

        self.game = SnakeGame(num_obstacles=4)
        self.fps = DEFAULT_FPS
        self.is_paused = False

        # Tải mô hình AI DRQN
        self.model = self._load_model()

        # Buffer trượt lưu chuỗi SEQ_LEN bước quan sát cho mạng LSTM
        self.history_buffer = deque(maxlen=SEQ_LEN)
        self._reset_history_buffer()

        # Dữ liệu dự đoán thời gian thực
        self.current_q_values = [0.0, 0.0, 0.0]
        self.chosen_action_idx = 0
        self.operator_names = ["Đi thẳng [1, 0, 0]", "Rẽ trái  [0, 1, 0]", "Rẽ phải  [0, 0, 1]"]

    def _reset_history_buffer(self):
        """Khởi tạo lại buffer với các vector 0."""
        self.history_buffer.clear()
        for _ in range(SEQ_LEN):
            self.history_buffer.append(np.zeros(FEATURE_DIM, dtype=np.float32))

    def _load_model(self):
        """Tải mô hình Keras v3 hoặc H5."""
        model = None
        if os.path.exists(MODEL_PATH):
            try:
                import keras
                model = keras.models.load_model(MODEL_PATH)
                print(f"[✓] Đã tải thành công mô hình DRQN từ: {MODEL_PATH}")
            except Exception as e:
                print(f"[!] Lỗi tải {MODEL_PATH}: {e}")

        if model is None and os.path.exists(MODEL_H5_PATH):
            try:
                import keras
                model = keras.models.load_model(MODEL_H5_PATH)
                print(f"[✓] Đã tải thành công mô hình DRQN từ: {MODEL_H5_PATH}")
            except Exception as e:
                print(f"[!] Lỗi tải {MODEL_H5_PATH}: {e}")

        if model is None:
            print("[!] Chưa tìm thấy file mô hình đã huấn luyện.")
            print("[*] Vui lòng chạy lệnh 'py -3 train_rl.py' để huấn luyện mô hình trước!")
        return model

    def _predict_q_values(self, input_batch: np.ndarray) -> list[float]:
        """Dự đoán giá trị Q-values cho 3 hành động an toàn."""
        if self.model is None:
            return [0.0, 0.0, 0.0]

        try:
            import torch
            with torch.no_grad():
                preds = self.model(input_batch, training=False)
                if hasattr(preds, 'detach'):
                    preds = preds.detach()
                if hasattr(preds, 'cpu'):
                    preds = preds.cpu()
                if hasattr(preds, 'numpy'):
                    q_vals = preds.numpy()[0]
                else:
                    q_vals = np.array(preds)[0]
                return [float(q) for q in q_vals]
        except Exception:
            pass

        try:
            preds = self.model(input_batch, training=False)
            if hasattr(preds, 'detach'):
                preds = preds.detach()
            if hasattr(preds, 'numpy'):
                q_vals = preds.numpy()[0]
            else:
                q_vals = np.array(preds)[0]
            return [float(q) for q in q_vals]
        except Exception:
            return [0.0, 0.0, 0.0]

    def _update_step(self):
        """Thực hiện một bước quan sát, suy luận Q-value và di chuyển của AI."""
        # 1. Trích xuất đặc trưng hiện tại
        current_feat = extract_features(
            head=self.game.head,
            current_dir=self.game.direction,
            snake_body=self.game.body,
            obstacles=self.game.obstacles,
            food=self.game.food
        )

        # 2. Cập nhật vào sliding window
        self.history_buffer.append(current_feat)
        seq_input = np.array(self.history_buffer, dtype=np.float32)  # (SEQ_LEN, FEATURE_DIM)
        input_batch = np.expand_dims(seq_input, axis=0)               # (1, SEQ_LEN, FEATURE_DIM)

        # 3. AI Inference: Dự đoán Q-values qua mạng DRQN LSTM
        q_vals = self._predict_q_values(input_batch)
        self.current_q_values = q_vals
        self.chosen_action_idx = int(np.argmax(q_vals))

        # 4. Ánh xạ thành toán tử tương đối
        actions_map = [ACTION_STRAIGHT, ACTION_LEFT, ACTION_RIGHT]
        action = actions_map[self.chosen_action_idx]

        # 5. Thực thi bước đi trong môi trường
        self.game.play_step(action)

    def _draw_all(self):
        """Vẽ toàn bộ bàn cờ và Dashboard thông số."""
        self.screen.fill(COLOR_BG_DARK)
        self._draw_board()
        self._draw_obstacles()
        self._draw_food()
        self._draw_snake()
        self._draw_sidebar()
        pygame.display.flip()

    def _draw_board(self):
        """Vẽ bàn cờ 14x14."""
        board_rect = pygame.Rect(0, 0, BOARD_WIDTH, BOARD_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_BOARD_BG, board_rect)
        for x in range(GRID_WIDTH + 1):
            px = x * CELL_SIZE
            pygame.draw.line(self.screen, COLOR_GRID_LINE, (px, 0), (px, BOARD_HEIGHT), 1)
        for y in range(GRID_HEIGHT + 1):
            py = y * CELL_SIZE
            pygame.draw.line(self.screen, COLOR_GRID_LINE, (0, py), (BOARD_WIDTH, py), 1)
        pygame.draw.rect(self.screen, (168, 85, 247), board_rect, 2)

    def _draw_obstacles(self):
        """Vẽ 4 chướng ngại vật cố định."""
        for obs in self.game.obstacles:
            rx = obs.x * CELL_SIZE
            ry = obs.y * CELL_SIZE
            rect = pygame.Rect(rx + 2, ry + 2, CELL_SIZE - 4, CELL_SIZE - 4)
            pygame.draw.rect(self.screen, COLOR_OBSTACLE, rect, border_radius=6)
            pygame.draw.rect(self.screen, COLOR_OBSTACLE_BORDER, rect, width=2, border_radius=6)
            pygame.draw.line(self.screen, COLOR_OBSTACLE_STRIPE, (rx + 8, ry + 8), (rx + CELL_SIZE - 8, ry + CELL_SIZE - 8), 2)
            pygame.draw.line(self.screen, COLOR_OBSTACLE_STRIPE, (rx + CELL_SIZE - 8, ry + 8), (rx + 8, ry + CELL_SIZE - 8), 2)

    def _draw_food(self):
        """Vẽ mồi với hiệu ứng phát sáng."""
        food = self.game.food
        if food is None:
            return
        cx = food.x * CELL_SIZE + CELL_SIZE // 2
        cy = food.y * CELL_SIZE + CELL_SIZE // 2
        radius = CELL_SIZE // 2 - 4
        glow_surface = pygame.Surface((CELL_SIZE * 2, CELL_SIZE * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surface, (239, 68, 68, 60), (CELL_SIZE, CELL_SIZE), radius + 6)
        self.screen.blit(glow_surface, (cx - CELL_SIZE, cy - CELL_SIZE))
        pygame.draw.circle(self.screen, COLOR_FOOD, (cx, cy), radius)
        pygame.draw.circle(self.screen, COLOR_FOOD_GLOW, (cx - 2, cy - 2), radius // 2)

    def _draw_snake(self):
        """Vẽ đầu và thân rắn."""
        full_snake = self.game.get_full_snake()
        if not full_snake:
            return
        total_segs = len(full_snake)
        for i in range(total_segs - 1, 0, -1):
            seg = full_snake[i]
            rx = seg.x * CELL_SIZE
            ry = seg.y * CELL_SIZE
            t = (i - 1) / max(total_segs - 1, 1)
            r = int(COLOR_SNAKE_BODY_START[0] * (1 - t) + COLOR_SNAKE_BODY_END[0] * t)
            g = int(COLOR_SNAKE_BODY_START[1] * (1 - t) + COLOR_SNAKE_BODY_END[1] * t)
            b = int(COLOR_SNAKE_BODY_START[2] * (1 - t) + COLOR_SNAKE_BODY_END[2] * t)
            rect = pygame.Rect(rx + 2, ry + 2, CELL_SIZE - 4, CELL_SIZE - 4)
            pygame.draw.rect(self.screen, (r, g, b), rect, border_radius=6)

        head = full_snake[0]
        hx = head.x * CELL_SIZE
        hy = head.y * CELL_SIZE
        head_rect = pygame.Rect(hx + 1, hy + 1, CELL_SIZE - 2, CELL_SIZE - 2)
        pygame.draw.rect(self.screen, COLOR_SNAKE_HEAD, head_rect, border_radius=8)

        # Mắt rắn
        d = self.game.direction
        eye_radius = 3
        pupil_radius = 2
        if d == DIR_RIGHT:
            eye1 = (hx + CELL_SIZE - 8, hy + 10)
            eye2 = (hx + CELL_SIZE - 8, hy + CELL_SIZE - 10)
        elif d == DIR_LEFT:
            eye1 = (hx + 8, hy + 10)
            eye2 = (hx + 8, hy + CELL_SIZE - 10)
        elif d == DIR_UP:
            eye1 = (hx + 10, hy + 8)
            eye2 = (hx + CELL_SIZE - 10, hy + 8)
        else:
            eye1 = (hx + 10, hy + CELL_SIZE - 8)
            eye2 = (hx + CELL_SIZE - 10, hy + CELL_SIZE - 8)

        pygame.draw.circle(self.screen, COLOR_SNAKE_EYES, eye1, eye_radius)
        pygame.draw.circle(self.screen, COLOR_SNAKE_EYES, eye2, eye_radius)
        pygame.draw.circle(self.screen, COLOR_SNAKE_PUPIL, eye1, pupil_radius)
        pygame.draw.circle(self.screen, COLOR_SNAKE_PUPIL, eye2, pupil_radius)

    def _draw_sidebar(self):
        """Vẽ Sidebar hiển thị thông số và Thanh Q-Values thời gian thực của DRQN."""
        sb_x = BOARD_WIDTH
        sb_rect = pygame.Rect(sb_x, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_SIDEBAR_BG, sb_rect)
        pygame.draw.line(self.screen, (30, 41, 59), (sb_x, 0), (sb_x, WINDOW_HEIGHT), 2)

        cur_y = 16
        # Tiêu đề
        title = self.font_title.render("AI SNAKE - DRQN (RL)", True, (168, 85, 247))
        self.screen.blit(title, (sb_x + 18, cur_y))
        cur_y += 28

        sub = self.font_small.render("Học Tăng Cường (Reinforcement Learning)", True, COLOR_TEXT_MUTED)
        self.screen.blit(sub, (sb_x + 18, cur_y))
        cur_y += 30

        # Panel Điểm số
        p_rect = pygame.Rect(sb_x + 16, cur_y, SIDEBAR_WIDTH - 32, 60)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, p_rect, border_radius=8)
        sc_lbl = self.font_small.render("ĐIỂM SỐ", True, COLOR_TEXT_MUTED)
        sc_val = self.font_title.render(str(self.game.score), True, (52, 211, 153))
        self.screen.blit(sc_lbl, (sb_x + 30, cur_y + 8))
        self.screen.blit(sc_val, (sb_x + 30, cur_y + 26))

        rec_lbl = self.font_small.render("KỶ LỤC", True, COLOR_TEXT_MUTED)
        rec_val = self.font_title.render(str(self.game.record_score), True, (251, 191, 36))
        self.screen.blit(rec_lbl, (sb_x + 180, cur_y + 8))
        self.screen.blit(rec_val, (sb_x + 180, cur_y + 26))
        cur_y += 72

        # Panel HÀM GIÁ TRỊ Q-VALUES CỦA AI DRQN
        q_panel = pygame.Rect(sb_x + 16, cur_y, SIDEBAR_WIDTH - 32, 160)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, q_panel, border_radius=8)
        q_title = self.font_sub.render("HÀM GIÁ TRỊ Q-VALUES (DRQN)", True, (56, 189, 248))
        self.screen.blit(q_title, (sb_x + 28, cur_y + 10))

        bar_y = cur_y + 36
        act_names = ["1. Thẳng", "2. Trái", "3. Phải"]

        # Chuẩn hóa để vẽ thanh bar tỷ lệ trực quan (dùng softmax tạm thời cho thanh trực quan)
        q_arr = np.array(self.current_q_values, dtype=np.float32)
        exp_q = np.exp(q_arr - np.max(q_arr))
        softmax_vis = exp_q / np.sum(exp_q)

        for i in range(3):
            q_val = self.current_q_values[i]
            bar_ratio = softmax_vis[i]
            is_chosen = (i == self.chosen_action_idx)
            lbl_color = (52, 211, 153) if is_chosen else COLOR_TEXT_WHITE
            
            # Hiển thị tên toán tử kèm giá trị Q thực tế
            name_txt = self.font_small.render(f"{act_names[i]}: Q={q_val:+.2f}", True, lbl_color)
            self.screen.blit(name_txt, (sb_x + 28, bar_y))

            # Thanh nền bar
            bg_bar = pygame.Rect(sb_x + 155, bar_y + 2, 145, 12)
            pygame.draw.rect(self.screen, (15, 23, 42), bg_bar, border_radius=4)
            
            # Thanh tiến trình fill
            fill_w = int(145 * min(max(bar_ratio, 0.0), 1.0))
            if fill_w > 0:
                bar_color = (16, 185, 129) if is_chosen else (100, 116, 139)
                fill_bar = pygame.Rect(sb_x + 155, bar_y + 2, fill_w, 12)
                pygame.draw.rect(self.screen, bar_color, fill_bar, border_radius=4)

            bar_y += 28

        cur_y += 172

        # Cảm biến va chạm 3 hướng
        dangers = ObstacleDetector.get_danger_sensors(
            self.game.head, self.game.direction, self.game.body,
            self.game.obstacles, self.game.grid_w, self.game.grid_h
        )
        sens_panel = pygame.Rect(sb_x + 16, cur_y, SIDEBAR_WIDTH - 32, 70)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, sens_panel, border_radius=8)
        s_title = self.font_sub.render("CẢM BIẾN VẬT CẢN (1 BƯỚC)", True, COLOR_TEXT_WHITE)
        self.screen.blit(s_title, (sb_x + 28, cur_y + 8))

        s_dirs = [("Thẳng", dangers['straight']), ("Trái", dangers['left']), ("Phải", dangers['right'])]
        for idx, (name, is_dan) in enumerate(s_dirs):
            bx = sb_x + 28 + idx * 95
            by = cur_y + 36
            dot_color = COLOR_DANGER if is_dan else COLOR_SAFE
            status_txt = "NGUY HIỂM" if is_dan else "AN TOÀN"
            pygame.draw.circle(self.screen, dot_color, (bx + 6, by + 6), 5)
            s_lbl = self.font_small.render(f"{name}: {status_txt}", True, dot_color)
            self.screen.blit(s_lbl, (bx + 16, by))

        cur_y += 82

        # Panel Thông tin & Phím tắt
        info_panel = pygame.Rect(sb_x + 16, cur_y, SIDEBAR_WIDTH - 32, 100)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, info_panel, border_radius=8)
        i_title = self.font_sub.render("BẢNG ĐIỀU KHIỂN & TRẠNG THÁI", True, (244, 63, 94))
        self.screen.blit(i_title, (sb_x + 28, cur_y + 8))

        state_str = "TẠM DỪNG (PAUSED)" if self.is_paused else f"ĐANG CHẠY (FPS: {self.fps})"
        st_lbl = self.font_small.render(f"Trạng thái : {state_str}", True, COLOR_TEXT_WHITE)
        self.screen.blit(st_lbl, (sb_x + 28, cur_y + 32))

        op_name = self.operator_names[self.chosen_action_idx]
        op_lbl = self.font_small.render(f"Toán tử AI : {op_name}", True, (52, 211, 153))
        self.screen.blit(op_lbl, (sb_x + 28, cur_y + 52))

        keys_lbl = self.font_small.render("Phím: [SPACE] Tạm dừng | [R] Chơi lại | [UP/DN] Tốc độ", True, COLOR_TEXT_MUTED)
        self.screen.blit(keys_lbl, (sb_x + 28, cur_y + 74))

    def run(self):
        """Vòng lặp chính của ứng dụng đồ họa."""
        running = True
        auto_restart_timer = 0

        while running:
            # 1. Xử lý sự kiện từ bàn phím và chuột
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.is_paused = not self.is_paused
                    elif event.key == pygame.K_r:
                        self.game.reset()
                        self._reset_history_buffer()
                        auto_restart_timer = 0
                    elif event.key == pygame.K_UP:
                        self.fps = min(self.fps + 5, 60)
                    elif event.key == pygame.K_DOWN:
                        self.fps = max(self.fps - 5, 2)

            # 2. Cập nhật bước đi của AI nếu không tạm dừng
            if not self.is_paused:
                if not self.game.is_game_over:
                    self._update_step()
                    auto_restart_timer = 0
                else:
                    # Tự động khởi động lại sau 1.5 giây khi chết
                    auto_restart_timer += 1
                    if auto_restart_timer > max(5, self.fps * 1.5):
                        self.game.reset()
                        self._reset_history_buffer()
                        auto_restart_timer = 0

            # 3. Vẽ lại khung hình
            self._draw_all()
            self.clock.tick(self.fps)

        pygame.quit()


def main():
    app = SnakeAIDRQNVisualizer()
    app.run()


if __name__ == "__main__":
    main()
