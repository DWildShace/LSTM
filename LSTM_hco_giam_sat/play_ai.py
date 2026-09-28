"""
Chương trình chơi Rắn săn mồi bằng Mô hình Trí tuệ Nhân tạo LSTM.
Mô hình AI tự quyết định hành động thông qua phân phối xác suất Softmax [Thẳng, Trái, Phải].
Hiển thị Dashboard đồ họa Pygame trực quan với thanh xác suất thời gian thực.
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

# Đường dẫn thư mục
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


class SnakeAILSTMVisualizer:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("AI Snake - Mô hình Deep Learning LSTM (2 Tầng) | 14x14")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        # Font chữ hỗ trợ tiếng Việt trên Windows
        self.font_title = pygame.font.SysFont('segoeui', 20, bold=True)
        self.font_sub = pygame.font.SysFont('segoeui', 15, bold=True)
        self.font_regular = pygame.font.SysFont('segoeui', 13)
        self.font_small = pygame.font.SysFont('segoeui', 11)
        self.font_bold = pygame.font.SysFont('segoeui', 13, bold=True)

        self.game = SnakeGame(num_obstacles=4)
        self.fps = DEFAULT_FPS
        self.is_paused = False

        # Tải mô hình AI đã huấn luyện
        self.model = self._load_model()

        # Buffer trượt lưu chuỗi SEQ_LEN bước quan sát
        self.history_buffer = deque(maxlen=SEQ_LEN)
        self._reset_history_buffer()

        # Dữ liệu dự đoán thời gian thực
        self.current_probs = [0.333, 0.333, 0.333]
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
                print(f"[✓] Đã tải thành công mô hình từ: {MODEL_PATH}")
            except Exception as e:
                print(f"[!] Lỗi tải {MODEL_PATH}: {e}")

        if model is None and os.path.exists(MODEL_H5_PATH):
            try:
                import keras
                model = keras.models.load_model(MODEL_H5_PATH)
                print(f"[✓] Đã tải thành công mô hình từ: {MODEL_H5_PATH}")
            except Exception as e:
                print(f"[!] Lỗi tải {MODEL_H5_PATH}: {e}")

        if model is None:
            print("[!] CẢNH BÁO: Chưa tìm thấy mô hình đã huấn luyện!")
            print("[*] Bạn hãy chạy 'py -3 train.py' trước để huấn luyện mô hình.")
        return model

    def run(self):
        """Vòng lặp chính."""
        running = True
        while running:
            running = self._handle_events()

            if not self.is_paused and not self.game.is_game_over:
                self._update_step()

            self._draw_all()
            self.clock.tick(self.fps)

        pygame.quit()
        sys.exit()

    def _handle_events(self) -> bool:
        """Xử lý sự kiện."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.is_paused = not self.is_paused
                elif event.key == pygame.K_r:
                    self.game.reset(randomize_obstacles=True)
                    self._reset_history_buffer()
                    self.current_probs = [0.333, 0.333, 0.333]
                elif event.key == pygame.K_UP:
                    self.fps = min(self.fps + 3, 60)
                elif event.key == pygame.K_DOWN:
                    self.fps = max(self.fps - 3, 2)
        return True

    def _predict_probabilities(self, input_batch) -> list:
        """Dự đoán phân phối xác suất Softmax an toàn tuyệt đối với torch no_grad và detach."""
        if self.model is None:
            return [1.0, 0.0, 0.0]

        try:
            import torch
            with torch.no_grad():
                preds = self.model(input_batch, training=False)
                if hasattr(preds, 'detach'):
                    preds = preds.detach()
                if hasattr(preds, 'cpu'):
                    preds = preds.cpu()
                if hasattr(preds, 'numpy'):
                    probs = preds.numpy()[0]
                else:
                    probs = np.array(preds)[0]
                return [float(p) for p in probs]
        except Exception:
            pass

        try:
            preds = self.model(input_batch, training=False)
            if hasattr(preds, 'detach'):
                preds = preds.detach()
            if hasattr(preds, 'numpy'):
                probs = preds.numpy()[0]
            else:
                probs = np.array(preds)[0]
            return [float(p) for p in probs]
        except Exception:
            return [1.0, 0.0, 0.0]

    def _update_step(self):
        """Thực hiện một bước dự đoán và di chuyển của AI."""
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

        # 3. AI Inference (Dự đoán qua mạng LSTM)
        probs = self._predict_probabilities(input_batch)
        self.current_probs = probs
        self.chosen_action_idx = int(np.argmax(probs))

        # 4. Ánh xạ thành toán tử tương đối
        actions_map = [ACTION_STRAIGHT, ACTION_LEFT, ACTION_RIGHT]
        action = actions_map[self.chosen_action_idx]

        # 5. Thực thi bước đi
        self.game.play_step(action)

    def _draw_all(self):
        """Vẽ toàn bộ bàn cờ và Dashboard."""
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
        pygame.draw.rect(self.screen, (56, 189, 248), board_rect, 2)

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
        """Vẽ mồi."""
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
        """Vẽ Sidebar hiển thị thông số và Thanh Softmax của AI."""
        sb_x = BOARD_WIDTH
        sb_rect = pygame.Rect(sb_x, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_SIDEBAR_BG, sb_rect)
        pygame.draw.line(self.screen, (30, 41, 59), (sb_x, 0), (sb_x, WINDOW_HEIGHT), 2)

        cur_y = 16
        # Tiêu đề
        title = self.font_title.render("AI SNAKE - LSTM", True, (56, 189, 248))
        self.screen.blit(title, (sb_x + 18, cur_y))
        cur_y += 28

        sub = self.font_small.render("Mô hình AI học từ Chuyên gia A* (Imitation)", True, COLOR_TEXT_MUTED)
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

        # Panel XÁC SUẤT SOFTMAX CỦA AI LSTM
        soft_panel = pygame.Rect(sb_x + 16, cur_y, SIDEBAR_WIDTH - 32, 160)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, soft_panel, border_radius=8)
        soft_title = self.font_sub.render("XÁC SUẤT SOFTMAX (AI LSTM)", True, (168, 85, 247))
        self.screen.blit(soft_title, (sb_x + 28, cur_y + 10))

        bar_y = cur_y + 36
        act_names = ["1. Thẳng", "2. Trái", "3. Phải"]
        for i in range(3):
            prob = self.current_probs[i]
            is_chosen = (i == self.chosen_action_idx)
            lbl_color = (52, 211, 153) if is_chosen else COLOR_TEXT_WHITE
            name_txt = self.font_small.render(f"{act_names[i]}: {prob*100:5.1f}%", True, lbl_color)
            self.screen.blit(name_txt, (sb_x + 28, bar_y))

            # Thanh nền bar
            bg_bar = pygame.Rect(sb_x + 140, bar_y + 2, 160, 12)
            pygame.draw.rect(self.screen, (15, 23, 42), bg_bar, border_radius=4)
            # Thanh tiến trình fill
            fill_w = int(160 * min(max(prob, 0.0), 1.0))
            if fill_w > 0:
                bar_color = (16, 185, 129) if is_chosen else (100, 116, 139)
                fill_bar = pygame.Rect(sb_x + 140, bar_y + 2, fill_w, 12)
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
            bx = sb_x + 30 + idx * 95
            by = cur_y + 34
            dot_color = COLOR_DANGER if is_dan else COLOR_SAFE
            status_text = "NGUY HIỂM" if is_dan else "AN TOÀN"
            pygame.draw.circle(self.screen, dot_color, (bx + 8, by + 10), 6)
            txt = self.font_small.render(f"{name}: {status_text}", True, dot_color)
            self.screen.blit(txt, (bx + 20, by + 2))

        cur_y += 82

        # Trạng thái game over
        if self.game.is_game_over:
            over_panel = pygame.Rect(sb_x + 16, cur_y, SIDEBAR_WIDTH - 32, 50)
            pygame.draw.rect(self.screen, (153, 27, 27), over_panel, border_radius=8)
            over_txt = self.font_sub.render("GAME OVER! Nhấn 'R' để thử lại", True, (254, 202, 202))
            self.screen.blit(over_txt, (sb_x + 24, cur_y + 14))
        cur_y += 60

        # Phím tắt
        help_txts = [
            "SPACE: Tạm dừng / Tiếp tục",
            "MŨI TÊN LÊN/XUỐNG: Điều chỉnh tốc độ (FPS)",
            "R: Tạo ván mới & Vật cản ngẫu nhiên"
        ]
        for ht in help_txts:
            t = self.font_small.render(ht, True, COLOR_TEXT_MUTED)
            self.screen.blit(t, (sb_x + 20, cur_y))
            cur_y += 18


if __name__ == "__main__":
    visualizer = SnakeAILSTMVisualizer()
    visualizer.run()
