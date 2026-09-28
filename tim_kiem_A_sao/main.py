"""
Chương trình chính: Trò chơi Rắn săn mồi AI tìm đường A* trên lưới 14x14
- 3 đến 4 vật cản cố định
- Thân rắn là vật cản động di chuyển theo từng bước
- 3 toán tử: Đi thẳng [1, 0, 0], Rẽ trái [0, 1, 0], Rẽ phải [0, 0, 1]
- Module cảm biến phát hiện vật cản (Trực diện, Trái, Phải)
- Hiển thị đường đi A* và bảng điều khiển Dashboard trực quan
"""

import sys
import pygame
from constants import (
    Point, GRID_WIDTH, GRID_HEIGHT, CELL_SIZE,
    BOARD_WIDTH, BOARD_HEIGHT, SIDEBAR_WIDTH, WINDOW_WIDTH, WINDOW_HEIGHT,
    DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT,
    COLOR_BG_DARK, COLOR_BOARD_BG, COLOR_GRID_LINE, COLOR_SIDEBAR_BG, COLOR_PANEL_BG,
    COLOR_SNAKE_HEAD, COLOR_SNAKE_BODY_START, COLOR_SNAKE_BODY_END,
    COLOR_SNAKE_EYES, COLOR_SNAKE_PUPIL,
    COLOR_FOOD, COLOR_FOOD_GLOW,
    COLOR_OBSTACLE, COLOR_OBSTACLE_BORDER, COLOR_OBSTACLE_STRIPE,
    COLOR_ASTAR_PATH, COLOR_TEXT_WHITE, COLOR_TEXT_MUTED, COLOR_TEXT_ACCENT,
    COLOR_DANGER, COLOR_SAFE,
    ACTION_STRAIGHT, ACTION_LEFT, ACTION_RIGHT,
    DEFAULT_FPS
)
from obstacle_detector import ObstacleDetector
from astar import AStarFinder
from snake_game import SnakeGame

class SnakeVisualizer:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("AI Snake - Bản đồ 14x14 | Thuật toán A* & Cảm biến vật cản")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        # Cài đặt phông chữ hỗ trợ tiếng Việt trên Windows
        self.font_title = pygame.font.SysFont('segoeui', 20, bold=True)
        self.font_sub = pygame.font.SysFont('segoeui', 15, bold=True)
        self.font_regular = pygame.font.SysFont('segoeui', 13)
        self.font_small = pygame.font.SysFont('segoeui', 11)
        self.font_bold = pygame.font.SysFont('segoeui', 13, bold=True)

        self.game = SnakeGame(num_obstacles=4)
        self.finder = AStarFinder(self.game.grid_w, self.game.grid_h)
        self.fps = DEFAULT_FPS
        self.is_paused = False
        self.show_astar_path = True
        self.auto_ai_mode = True  # True: AI A*, False: Điều khiển thủ công

        # Dữ liệu từ bước tìm đường A* gần nhất
        self.current_action = ACTION_STRAIGHT
        self.next_pos = None
        self.astar_path = []
        self.astar_status = "Đang khởi tạo..."

    def run(self):
        """Vòng lặp chính của trò chơi."""
        running = True
        while running:
            # 1. Xử lý sự kiện từ bàn phím và chuột
            running = self._handle_events()

            # 2. Cập nhật logic game nếu không pause
            if not self.is_paused and not self.game.is_game_over:
                self._update_step()

            # 3. Vẽ toàn bộ giao diện
            self._draw_all()

            # 4. Giới hạn tốc độ khung hình
            self.clock.tick(self.fps)

        pygame.quit()
        sys.exit()

    def _handle_events(self) -> bool:
        """Xử lý sự kiện bàn phím."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                # Phím Space: Tạm dừng / Tiếp tục
                if event.key == pygame.K_SPACE:
                    self.is_paused = not self.is_paused

                # Phím R: Reset trò chơi và sinh vật cản mới
                elif event.key == pygame.K_r:
                    self.game.reset(randomize_obstacles=True)
                    self.astar_path = []
                    self.astar_status = "Đã tạo bản đồ mới"

                # Phím P: Bật / Tắt vẽ đường đi A*
                elif event.key == pygame.K_p:
                    self.show_astar_path = not self.show_astar_path

                # Phím M: Chuyển đổi giữa chế độ AI và Tự lái
                elif event.key == pygame.K_m:
                    self.auto_ai_mode = not self.auto_ai_mode

                # Phím Mũi tên lên / xuống: Tăng / Giảm tốc độ FPS
                elif event.key == pygame.K_UP:
                    self.fps = min(60, self.fps + 3)
                elif event.key == pygame.K_DOWN:
                    self.fps = max(2, self.fps - 3)

                # Điều khiển thủ công bằng bàn phím (khi auto_ai_mode = False)
                elif not self.auto_ai_mode and not self.game.is_game_over:
                    if event.key == pygame.K_LEFT:
                        self.game.play_step(ACTION_LEFT)
                    elif event.key == pygame.K_RIGHT:
                        self.game.play_step(ACTION_RIGHT)
                    elif event.key == pygame.K_UP or event.key == pygame.K_w:
                        self.game.play_step(ACTION_STRAIGHT)

        return True

    def _update_step(self):
        """Thực thi một bước di chuyển của AI A*."""
        if not self.auto_ai_mode:
            return

        if self.game.food is None:
            return

        # 1. Gọi thuật toán A* tối ưu CTDL để tính toán đường đi và 3 toán tử
        action, next_pos, full_path, status_text = self.finder.get_next_action(
            head=self.game.head,
            current_dir=self.game.direction,
            food=self.game.food,
            obstacles=self.game.obstacles,
            snake_body=self.game.body
        )

        self.current_action = action
        self.next_pos = next_pos
        self.astar_path = full_path
        self.astar_status = status_text

        # 2. Thực thi toán tử vừa chọn trong trò chơi
        reward, game_over, score = self.game.play_step(action)

        # 3. Nếu game over ở chế độ AI, tự động reset sau giây lát
        if game_over:
            self.astar_status = "Đã va chạm! Đang khởi động lại..."

    def _draw_all(self):
        """Vẽ toàn bộ bàn cờ và bảng điều khiển."""
        self.screen.fill(COLOR_BG_DARK)

        # 1. Vẽ bản đồ trò chơi 14x14
        self._draw_board()

        # 2. Vẽ bảng điều khiển AI bên phải
        self._draw_sidebar()

        pygame.display.flip()

    def _draw_board(self):
        """Vẽ lưới 14x14, vật cản, mồi, thân rắn và đường A*."""
        # Nền bản đồ
        board_rect = pygame.Rect(0, 0, BOARD_WIDTH, BOARD_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_BOARD_BG, board_rect)

        # Các đường kẻ lưới 14x14
        for x in range(GRID_WIDTH + 1):
            px = x * CELL_SIZE
            pygame.draw.line(self.screen, COLOR_GRID_LINE, (px, 0), (px, BOARD_HEIGHT), 1)
        for y in range(GRID_HEIGHT + 1):
            py = y * CELL_SIZE
            pygame.draw.line(self.screen, COLOR_GRID_LINE, (0, py), (BOARD_WIDTH, py), 1)

        # Vẽ đường đi A* dự kiến (nếu bật chế độ vẽ)
        if self.show_astar_path and self.astar_path:
            self._draw_astar_path()

        # Vẽ 3-4 vật cản cố định
        for obs in self.game.obstacles:
            self._draw_obstacle(obs)

        # Vẽ mồi
        if self.game.food:
            self._draw_food(self.game.food)

        # Vẽ thân rắn (vật cản động)
        self._draw_snake()

        # Đường viền phân cách bản đồ và sidebar
        pygame.draw.line(self.screen, (51, 65, 85), (BOARD_WIDTH, 0), (BOARD_WIDTH, BOARD_HEIGHT), 2)

        # Thông báo Game Over
        if self.game.is_game_over:
            overlay = pygame.Surface((BOARD_WIDTH, BOARD_HEIGHT), pygame.SRCALPHA)
            overlay.fill((15, 23, 42, 200))
            self.screen.blit(overlay, (0, 0))

            t_over = self.font_title.render("TRÒ CHƠI KẾT THÚC!", True, COLOR_DANGER)
            t_sub = self.font_sub.render(f"Điểm đạt được: {self.game.score} | Nhấn 'R' để chơi lại", True, COLOR_TEXT_WHITE)
            self.screen.blit(t_over, (BOARD_WIDTH // 2 - t_over.get_width() // 2, BOARD_HEIGHT // 2 - 30))
            self.screen.blit(t_sub, (BOARD_WIDTH // 2 - t_sub.get_width() // 2, BOARD_HEIGHT // 2 + 15))

    def _draw_obstacle(self, pt: Point):
        """Vẽ một ô vật cản cố định với họa tiết cảnh báo sọc gạch chéo."""
        x = pt.x * CELL_SIZE
        y = pt.y * CELL_SIZE
        rect = pygame.Rect(x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4)

        # Nền vật cản
        pygame.draw.rect(self.screen, COLOR_OBSTACLE, rect, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_OBSTACLE_BORDER, rect, width=2, border_radius=6)

        # Các vạch sọc chéo cảnh báo
        pygame.draw.line(self.screen, COLOR_OBSTACLE_STRIPE, (x + 8, y + CELL_SIZE - 6), (x + CELL_SIZE - 6, y + 8), 3)
        pygame.draw.line(self.screen, COLOR_OBSTACLE_STRIPE, (x + 16, y + CELL_SIZE - 6), (x + CELL_SIZE - 6, y + 16), 3)
        pygame.draw.line(self.screen, COLOR_OBSTACLE_STRIPE, (x + 6, y + CELL_SIZE - 16), (x + CELL_SIZE - 16, y + 6), 3)

    def _draw_food(self, pt: Point):
        """Vẽ mồi với hiệu ứng ánh sáng phát quang."""
        center_x = pt.x * CELL_SIZE + CELL_SIZE // 2
        center_y = pt.y * CELL_SIZE + CELL_SIZE // 2
        radius = CELL_SIZE // 2 - 5

        # Vòng hào quang mồi
        pygame.draw.circle(self.screen, COLOR_FOOD_GLOW, (center_x, center_y), radius + 2, width=1)
        # Viên mồi chính
        pygame.draw.circle(self.screen, COLOR_FOOD, (center_x, center_y), radius)
        # Đốm sáng phản chiếu
        pygame.draw.circle(self.screen, (255, 255, 255), (center_x - 3, center_y - 3), 3)

    def _draw_snake(self):
        """Vẽ rắn với đầu, mắt nhìn theo hướng và thân có gradient."""
        full_snake = self.game.get_full_snake()
        n = len(full_snake)

        # Vẽ thân rắn trước (từ đuôi lên đầu)
        for i in range(len(self.game.body) - 1, -1, -1):
            seg = self.game.body[i]
            # Tính gradient màu sắc từ xanh sáng tới xanh đậm
            ratio = i / max(1, len(self.game.body))
            r = int(COLOR_SNAKE_BODY_START[0] * (1 - ratio) + COLOR_SNAKE_BODY_END[0] * ratio)
            g = int(COLOR_SNAKE_BODY_START[1] * (1 - ratio) + COLOR_SNAKE_BODY_END[1] * ratio)
            b = int(COLOR_SNAKE_BODY_START[2] * (1 - ratio) + COLOR_SNAKE_BODY_END[2] * ratio)
            color = (r, g, b)

            x = seg.x * CELL_SIZE
            y = seg.y * CELL_SIZE
            rect = pygame.Rect(x + 3, y + 3, CELL_SIZE - 6, CELL_SIZE - 6)
            pygame.draw.rect(self.screen, color, rect, border_radius=7)

            # Số thứ tự đốt thân (nhỏ ở giữa nếu ô đủ rộng)
            if i == len(self.game.body) - 1:
                # Đuôi rắn có dấu chấm nhỏ
                pygame.draw.circle(self.screen, (255, 255, 255, 120), (x + CELL_SIZE // 2, y + CELL_SIZE // 2), 3)

        # Vẽ đầu rắn
        head = self.game.head
        hx = head.x * CELL_SIZE
        hy = head.y * CELL_SIZE
        head_rect = pygame.Rect(hx + 2, hy + 2, CELL_SIZE - 4, CELL_SIZE - 4)
        pygame.draw.rect(self.screen, COLOR_SNAKE_HEAD, head_rect, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255), head_rect, width=1, border_radius=10)

        # Vẽ 2 mắt rắn nhìn theo hướng di chuyển hiện tại
        d = self.game.direction
        cx = hx + CELL_SIZE // 2
        cy = hy + CELL_SIZE // 2

        eye_offset = 7
        pupil_offset = 2

        if d == DIR_RIGHT:
            eye1 = (cx + 5, cy - eye_offset)
            eye2 = (cx + 5, cy + eye_offset)
            pupil1 = (eye1[0] + pupil_offset, eye1[1])
            pupil2 = (eye2[0] + pupil_offset, eye2[1])
        elif d == DIR_LEFT:
            eye1 = (cx - 5, cy - eye_offset)
            eye2 = (cx - 5, cy + eye_offset)
            pupil1 = (eye1[0] - pupil_offset, eye1[1])
            pupil2 = (eye2[0] - pupil_offset, eye2[1])
        elif d == DIR_UP:
            eye1 = (cx - eye_offset, cy - 5)
            eye2 = (cx + eye_offset, cy - 5)
            pupil1 = (eye1[0], eye1[1] - pupil_offset)
            pupil2 = (eye2[0], eye2[1] - pupil_offset)
        else: # DIR_DOWN
            eye1 = (cx - eye_offset, cy + 5)
            eye2 = (cx + eye_offset, cy + 5)
            pupil1 = (eye1[0], eye1[1] + pupil_offset)
            pupil2 = (eye2[0], eye2[1] + pupil_offset)

        pygame.draw.circle(self.screen, COLOR_SNAKE_EYES, eye1, 4)
        pygame.draw.circle(self.screen, COLOR_SNAKE_EYES, eye2, 4)
        pygame.draw.circle(self.screen, COLOR_SNAKE_PUPIL, pupil1, 2)
        pygame.draw.circle(self.screen, COLOR_SNAKE_PUPIL, pupil2, 2)

    def _draw_astar_path(self):
        """Vẽ các bước đi dự kiến của thuật toán A* trên lưới."""
        points = [self.game.head] + self.astar_path
        if len(points) < 2:
            return

        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            c1 = (p1.x * CELL_SIZE + CELL_SIZE // 2, p1.y * CELL_SIZE + CELL_SIZE // 2)
            c2 = (p2.x * CELL_SIZE + CELL_SIZE // 2, p2.y * CELL_SIZE + CELL_SIZE // 2)

            pygame.draw.line(self.screen, COLOR_ASTAR_PATH, c1, c2, 3)
            # Chấm tròn nhỏ tại các bước trung gian
            pygame.draw.circle(self.screen, COLOR_ASTAR_PATH, c2, 4)

    def _draw_sidebar(self):
        """Vẽ thanh thông tin bên phải (Dashboard AI, Cảm biến, Toán tử)."""
        sidebar_rect = pygame.Rect(BOARD_WIDTH, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_SIDEBAR_BG, sidebar_rect)

        start_x = BOARD_WIDTH + 18
        cur_y = 16

        # Header Title
        title_surf = self.font_title.render("RẮN SĂN MỒI A*", True, COLOR_TEXT_ACCENT)
        self.screen.blit(title_surf, (start_x, cur_y))
        cur_y += 26
        sub_title = self.font_small.render("Bản đồ 14x14 | 4 Vật cản cố định & Thân động", True, COLOR_TEXT_MUTED)
        self.screen.blit(sub_title, (start_x, cur_y))
        cur_y += 24

        # Card 1: Điểm số & Kỷ lục
        cur_y = self._draw_card_stats(start_x, cur_y)

        # Card 2: 3 Toán tử chính (Đi thẳng, Rẽ trái, Rẽ phải)
        cur_y = self._draw_card_operators(start_x, cur_y)

        # Card 3: Module phát hiện vật cản (Sensors)
        cur_y = self._draw_card_sensors(start_x, cur_y)

        # Card 4: Trạng thái thuật toán A*
        cur_y = self._draw_card_astar_status(start_x, cur_y)

        # Card 5: Hướng dẫn phím bấm
        self._draw_card_controls(start_x, cur_y)

    def _draw_card_stats(self, x: int, y: int) -> int:
        card_w = SIDEBAR_WIDTH - 36
        card_h = 60
        rect = pygame.Rect(x, y, card_w, card_h)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, rect, border_radius=8)

        # Điểm
        t_score_lbl = self.font_small.render("ĐIỂM HIỆN TẠI", True, COLOR_TEXT_MUTED)
        t_score_val = self.font_sub.render(str(self.game.score), True, COLOR_SAFE)
        self.screen.blit(t_score_lbl, (x + 14, y + 10))
        self.screen.blit(t_score_val, (x + 14, y + 28))

        # Kỷ lục
        t_rec_lbl = self.font_small.render("KỶ LỤC CAO NHẤT", True, COLOR_TEXT_MUTED)
        t_rec_val = self.font_sub.render(str(self.game.record_score), True, COLOR_TEXT_ACCENT)
        self.screen.blit(t_rec_lbl, (x + 110, y + 10))
        self.screen.blit(t_rec_val, (x + 110, y + 28))

        # Chiều dài
        length = len(self.game.get_full_snake())
        t_len_lbl = self.font_small.render("CHIỀU DÀI", True, COLOR_TEXT_MUTED)
        t_len_val = self.font_sub.render(f"{length}", True, COLOR_TEXT_WHITE)
        self.screen.blit(t_len_lbl, (x + 225, y + 10))
        self.screen.blit(t_len_val, (x + 225, y + 28))

        return y + card_h + 12

    def _draw_card_operators(self, x: int, y: int) -> int:
        """Hiển thị trực quan 3 toán tử chính: Đi thẳng, Rẽ trái, Rẽ phải."""
        card_w = SIDEBAR_WIDTH - 36
        card_h = 92
        rect = pygame.Rect(x, y, card_w, card_h)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, rect, border_radius=8)

        header = self.font_sub.render("3 TOÁN TỬ CHÍNH (ACTIONS)", True, COLOR_TEXT_ACCENT)
        self.screen.blit(header, (x + 12, y + 8))

        # 3 Toán tử: [Đi thẳng, Rẽ trái, Rẽ phải]
        operators = [
            ("Đi thẳng", ACTION_STRAIGHT, "[1, 0, 0]", (self.current_action == ACTION_STRAIGHT)),
            ("Rẽ trái", ACTION_LEFT, "[0, 1, 0]", (self.current_action == ACTION_LEFT)),
            ("Rẽ phải", ACTION_RIGHT, "[0, 0, 1]", (self.current_action == ACTION_RIGHT))
        ]

        btn_w = (card_w - 24 - 12) // 3
        btn_y = y + 36

        for i, (name, action_code, vec_str, is_active) in enumerate(operators):
            bx = x + 12 + i * (btn_w + 6)
            brect = pygame.Rect(bx, btn_y, btn_w, 44)

            # Màu nút: sáng xanh nếu đang kích hoạt, mờ nếu không
            if is_active:
                bg_col = (16, 185, 129, 60)
                border_col = COLOR_SAFE
                text_col = COLOR_TEXT_WHITE
            else:
                bg_col = (20, 27, 45)
                border_col = (51, 65, 85)
                text_col = COLOR_TEXT_MUTED

            pygame.draw.rect(self.screen, bg_col, brect, border_radius=6)
            pygame.draw.rect(self.screen, border_col, brect, width=2 if is_active else 1, border_radius=6)

            # Text tên toán tử và vector nhị phân
            txt1 = self.font_small.render(name, True, text_col)
            txt2 = self.font_small.render(vec_str, True, COLOR_TEXT_ACCENT if is_active else COLOR_TEXT_MUTED)

            self.screen.blit(txt1, (bx + btn_w // 2 - txt1.get_width() // 2, btn_y + 6))
            self.screen.blit(txt2, (bx + btn_w // 2 - txt2.get_width() // 2, btn_y + 24))

        return y + card_h + 12

    def _draw_card_sensors(self, x: int, y: int) -> int:
        """Hiển thị Module cảm biến phát hiện vật cản (Trực diện, Trái, Phải)."""
        card_w = SIDEBAR_WIDTH - 36
        card_h = 110
        rect = pygame.Rect(x, y, card_w, card_h)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, rect, border_radius=8)

        header = self.font_sub.render("MODULE PHÁT HIỆN VẬT CẢN", True, COLOR_TEXT_ACCENT)
        self.screen.blit(header, (x + 12, y + 8))

        # Lấy trạng thái cảm biến hiện tại
        dangers = ObstacleDetector.get_danger_sensors(
            self.game.head, self.game.direction,
            self.game.body, self.game.obstacles,
            self.game.grid_w, self.game.grid_h
        )

        sensors = [
            ("Phía trước (Straight)", dangers['straight']),
            ("Bên trái (Left)", dangers['left']),
            ("Bên phải (Right)", dangers['right']),
        ]

        sy = y + 34
        for name, is_danger in sensors:
            lbl = self.font_regular.render(name, True, COLOR_TEXT_WHITE)
            self.screen.blit(lbl, (x + 14, sy))

            # Badge trạng thái: NGUY HIỂM / AN TOÀN
            badge_text = "NGUY HIỂM" if is_danger else "AN TOÀN"
            badge_color = COLOR_DANGER if is_danger else COLOR_SAFE
            badge_surf = self.font_small.render(badge_text, True, badge_color)
            self.screen.blit(badge_surf, (x + card_w - badge_surf.get_width() - 14, sy))

            # Đèn LED tròn
            led_x = x + card_w - badge_surf.get_width() - 24
            pygame.draw.circle(self.screen, badge_color, (led_x, sy + 7), 4)

            sy += 22

        return y + card_h + 12

    def _draw_card_astar_status(self, x: int, y: int) -> int:
        """Hiển thị trạng thái thuật toán A*, tốc độ tính toán vi giây và cấu trúc dữ liệu."""
        card_w = SIDEBAR_WIDTH - 36
        card_h = 100
        rect = pygame.Rect(x, y, card_w, card_h)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, rect, border_radius=8)

        header = self.font_sub.render("THUẬT TOÁN A* TỐI ƯU CTDL", True, COLOR_TEXT_ACCENT)
        self.screen.blit(header, (x + 12, y + 6))

        # Trạng thái hiện tại
        t_status = self.font_regular.render(f"Trạng thái: {self.astar_status}", True, COLOR_TEXT_WHITE)
        self.screen.blit(t_status, (x + 14, y + 26))

        # Chiều dài đường A* và khoảng cách Manhattan
        if self.game.food:
            dist = abs(self.game.head.x - self.game.food.x) + abs(self.game.head.y - self.game.food.y)
        else:
            dist = 0

        path_len = len(self.astar_path)
        info_str = f"Độ dài A*: {path_len} bước  |  Manhattan: {dist}"
        t_info = self.font_small.render(info_str, True, COLOR_TEXT_MUTED)
        self.screen.blit(t_info, (x + 14, y + 46))

        # Tốc độ tính toán (Microseconds) & Số node duyệt
        speed_str = f"Tốc độ: {self.finder.last_compute_time_us:.1f} µs  |  Đã duyệt: {self.finder.last_nodes_explored} nodes"
        t_speed = self.font_small.render(speed_str, True, COLOR_SAFE)
        self.screen.blit(t_speed, (x + 14, y + 63))

        # Các kỹ thuật CTDL áp dụng
        dsa_str = "CTDL: Bitboard + Epoch O(1) + Space-Time"
        t_dsa = self.font_small.render(dsa_str, True, (217, 119, 6))
        self.screen.blit(t_dsa, (x + 14, y + 80))

        return y + card_h + 10

    def _draw_card_controls(self, x: int, y: int):
        """Hiển thị hướng dẫn phím tắt và chế độ."""
        card_w = SIDEBAR_WIDTH - 36
        card_h = 90
        rect = pygame.Rect(x, y, card_w, card_h)
        pygame.draw.rect(self.screen, COLOR_PANEL_BG, rect, border_radius=8)

        header = self.font_sub.render(f"ĐIỀU KHIỂN (FPS: {self.fps})", True, COLOR_TEXT_ACCENT)
        self.screen.blit(header, (x + 12, y + 8))

        controls = [
            ("SPACE", "Tạm dừng / Tiếp tục"),
            ("UP / DOWN", "Tăng / Giảm tốc độ FPS"),
            ("R", "Reset & Tạo 4 vật cản mới"),
            ("P", "Bật/Tắt đường A*"),
            ("M", f"Chế độ: {'AI A*' if self.auto_ai_mode else 'Thủ công'}")
        ]

        cy = y + 30
        for key_name, desc in controls:
            k_surf = self.font_bold.render(f"[{key_name}]", True, COLOR_SAFE)
            d_surf = self.font_small.render(desc, True, COLOR_TEXT_MUTED)
            self.screen.blit(k_surf, (x + 14, cy))
            self.screen.blit(d_surf, (x + 85, cy + 1))
            cy += 14

if __name__ == "__main__":
    app = SnakeVisualizer()
    app.run()
