"""
Môi trường trò chơi Rắn săn mồi (Snake Game) kích thước lưới 14x14 không có vật cản.
Hỗ trợ chế độ đồ hoạ với Pygame và chế độ không giao diện (headless) để huấn luyện tốc độ cao.
"""

import random
from collections import deque
from enum import Enum
import numpy as np
import pygame


class Direction(Enum):
    UP = (0, -1)
    RIGHT = (1, 0)
    DOWN = (0, 1)
    LEFT = (-1, 0)


# Thứ tự vòng quay theo chiều kim đồng hồ: UP -> RIGHT -> DOWN -> LEFT
CLOCKWISE_DIRECTIONS = [
    Direction.UP,
    Direction.RIGHT,
    Direction.DOWN,
    Direction.LEFT,
]


class SnakeGame:
    def __init__(self, grid_size=14, cell_size=30, render_mode=False):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.render_mode = render_mode

        self.width = self.grid_size * self.cell_size
        self.height = self.grid_size * self.cell_size + 60  # Dành 60px cho bảng thông tin

        self.display = None
        self.clock = None
        self.font = None
        self.small_font = None

        if self.render_mode:
            self._init_pygame()

        self.reset()

    def _init_pygame(self):
        pygame.init()
        pygame.display.set_caption(f"Rắn Săn Mồi AI (DRQN / LSTM) - Bản đồ {self.grid_size}x{self.grid_size}")
        self.display = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont("Arial", 22, bold=True)
            self.small_font = pygame.font.SysFont("Arial", 16)
        except Exception:
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)

    def enable_render(self):
        if not self.render_mode:
            self.render_mode = True
            self._init_pygame()

    def reset(self):
        """Khởi tạo lại trạng thái trò chơi"""
        # Đặt rắn ở giữa bản đồ
        center_x = self.grid_size // 2
        center_y = self.grid_size // 2

        self.direction = Direction.RIGHT
        self.snake = [
            (center_x, center_y),
            (center_x - 1, center_y),
            (center_x - 2, center_y),
        ]
        self.snake_set = set(self.snake)
        self.head = self.snake[0]
        self.score = 0
        self.steps = 0
        self.steps_since_last_food = 0
        self.food = None
        self._place_food()

        return self.get_state()

    def _place_food(self):
        """Đặt mồi ngẫu nhiên vào ô trống chưa bị rắn chiếm (O(1) set lookup)"""
        empty_cells = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in self.snake_set
        ]
        if empty_cells:
            self.food = random.choice(empty_cells)
        else:
            # Rắn đã thắng toàn bộ màn chơi (chiếm hết 14x14)
            self.food = None

    def is_collision(self, pt=None):
        """
        Kiểm tra điểm pt có va chạm vào tường hoặc thân rắn không.
        - Tường: Ngoài biên bản đồ 14x14 -> Va chạm.
        - Thân rắn: Nếu pt trùng bất kỳ đốt thân nào (trừ đầu hiện tại self.snake[0]).
        - Đuôi: Nếu pt trùng ô đuôi và không có mồi, thì bước tiếp theo đuôi sẽ co lại nên an toàn.
        """
        if pt is None:
            pt = self.head

        # 1. Va chạm biên tường (bản đồ 14x14)
        if pt[0] < 0 or pt[0] >= self.grid_size or pt[1] < 0 or pt[1] >= self.grid_size:
            return True

        # 2. Va chạm thân rắn (bất kỳ ô nào thuộc thân rắn trừ chính đầu hiện tại)
        if pt in self.snake_set:
            if pt == self.snake[0]:
                return False
            return True

        return False

    def get_action_mask(self):
        """
        Trả về danh sách 3 giá trị boolean [an_toàn_thẳng, an_toàn_trái, an_toàn_phải] (Action Masking).
        True biểu thị hành động đi vào ô an toàn (không chết trong 1 bước).
        """
        head = self.head
        dir_idx = CLOCKWISE_DIRECTIONS.index(self.direction)
        dir_straight = CLOCKWISE_DIRECTIONS[dir_idx]
        dir_left = CLOCKWISE_DIRECTIONS[(dir_idx - 1) % 4]
        dir_right = CLOCKWISE_DIRECTIONS[(dir_idx + 1) % 4]

        pt_s = (head[0] + dir_straight.value[0], head[1] + dir_straight.value[1])
        pt_l = (head[0] + dir_left.value[0], head[1] + dir_left.value[1])
        pt_r = (head[0] + dir_right.value[0], head[1] + dir_right.value[1])

        return [
            not self.is_collision(pt_s),  # Action 0: Đi thẳng
            not self.is_collision(pt_l),  # Action 1: Rẽ trái
            not self.is_collision(pt_r),  # Action 2: Rẽ phải
        ]

    def _get_distance_to_obstacle(self, start_pt, dx, dy):
        """Đo khoảng cách từ start_pt theo hướng (dx, dy) tới vật cản gần nhất (O(1) lookup)"""
        x, y = start_pt
        dist = 0
        while True:
            dist += 1
            x += dx
            y += dy
            if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size or (x, y) in self.snake_set:
                break
        return dist

    def _calculate_flood_fill_ratio(self, start_pt):
        """
        Tính tỉ lệ diện tích trống có thể di chuyển từ start_pt bằng BFS.
        Tối ưu hoá: Khi rắn ngắn (< 10 ô) không thể tạo vòng lặp tự nhốt, bỏ qua BFS để tăng tốc.
        """
        if self.is_collision(start_pt):
            return 0.0

        # Rắn ngắn hơn 10 ô không thể tạo thành hình hộp khép kín tự bẫy trên lưới 14x14
        if len(self.snake) < 10:
            return 1.0

        visited = self.snake_set.copy()
        visited.add(start_pt)
        queue = deque([start_pt])
        count = 0
        max_search = min(35, len(self.snake) + 5)

        while queue and count < max_search:
            cx, cy = queue.popleft()
            count += 1
            for nx, ny in [(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)]:
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))

        return min(count / max_search, 1.0)

    def get_state(self):
        """
        Trích xuất vector trạng thái (State) 22 chiều phù hợp với mô hình LSTM:
        - 1-3: Nguy hiểm 1 bước (Thẳng, Phải, Trái)
        - 4-6: Nguy hiểm 2 bước (Thẳng, Phải, Trái)
        - 7-10: Hướng di chuyển hiện tại (Lên, Phải, Xuống, Trái)
        - 11-14: Hướng thức ăn tương đối (Thẳng, Phải, Trái, Sau lưng)
        - 15-16: Khoảng cách chuẩn hoá tới mồi (delta X, delta Y)
        - 17: Khoảng cách Euclidean chuẩn hoá tới mồi
        - 18: Độ dài thân rắn chuẩn hoá (len / 196)
        - 19-21: Khoảng cách raycast tới vật cản (Thẳng, Phải, Trái) chuẩn hoá [0, 1]
        - 22: Tỉ lệ diện tích trống (Flood-fill) phía trước để tránh bẫy
        """
        head = self.head

        # Xác định 3 điểm lân cận tương đối: Đi thẳng, Rẽ phải, Rẽ trái
        dir_idx = CLOCKWISE_DIRECTIONS.index(self.direction)
        dir_straight = CLOCKWISE_DIRECTIONS[dir_idx]
        dir_right = CLOCKWISE_DIRECTIONS[(dir_idx + 1) % 4]
        dir_left = CLOCKWISE_DIRECTIONS[(dir_idx - 1) % 4]

        # Điểm 1 bước
        pt_straight_1 = (head[0] + dir_straight.value[0], head[1] + dir_straight.value[1])
        pt_right_1 = (head[0] + dir_right.value[0], head[1] + dir_right.value[1])
        pt_left_1 = (head[0] + dir_left.value[0], head[1] + dir_left.value[1])

        # Điểm 2 bước
        pt_straight_2 = (head[0] + 2 * dir_straight.value[0], head[1] + 2 * dir_straight.value[1])
        pt_right_2 = (head[0] + 2 * dir_right.value[0], head[1] + 2 * dir_right.value[1])
        pt_left_2 = (head[0] + 2 * dir_left.value[0], head[1] + 2 * dir_left.value[1])

        # Nguy hiểm 1 bước & 2 bước
        danger_s1 = self.is_collision(pt_straight_1)
        danger_r1 = self.is_collision(pt_right_1)
        danger_l1 = self.is_collision(pt_left_1)

        danger_s2 = self.is_collision(pt_straight_2)
        danger_r2 = self.is_collision(pt_right_2)
        danger_l2 = self.is_collision(pt_left_2)

        # Hướng di chuyển toàn cục
        dir_up = self.direction == Direction.UP
        dir_right_flag = self.direction == Direction.RIGHT
        dir_down = self.direction == Direction.DOWN
        dir_left_flag = self.direction == Direction.LEFT

        # Vị trí thức ăn
        if self.food is not None:
            food_x, food_y = self.food
            dx = (food_x - head[0]) / self.grid_size
            dy = (food_y - head[1]) / self.grid_size
            euclidean_dist = np.sqrt((food_x - head[0]) ** 2 + (food_y - head[1]) ** 2) / (self.grid_size * 1.414)

            # Vị trí mồi theo hệ quy chiếu tương đối của đầu rắn
            # Vector đầu hướng đi: (dir_straight.value[0], dir_straight.value[1])
            vx, vy = dir_straight.value
            rx, ry = dir_right.value

            vec_to_food = (food_x - head[0], food_y - head[1])
            dot_front = vec_to_food[0] * vx + vec_to_food[1] * vy
            dot_right = vec_to_food[0] * rx + vec_to_food[1] * ry

            food_straight = dot_front > 0
            food_behind = dot_front < 0
            food_is_right = dot_right > 0
            food_is_left = dot_right < 0
        else:
            dx, dy, euclidean_dist = 0.0, 0.0, 0.0
            food_straight, food_behind, food_is_right, food_is_left = False, False, False, False

        # Tỉ lệ chiều dài rắn trên tổng số ô bản đồ 14x14 = 196
        snake_len_ratio = len(self.snake) / (self.grid_size * self.grid_size)

        # Khoảng cách raycast tới tường hoặc thân rắn
        ray_straight = self._get_distance_to_obstacle(head, dir_straight.value[0], dir_straight.value[1]) / self.grid_size
        ray_right = self._get_distance_to_obstacle(head, dir_right.value[0], dir_right.value[1]) / self.grid_size
        ray_left = self._get_distance_to_obstacle(head, dir_left.value[0], dir_left.value[1]) / self.grid_size

        # Không gian an toàn trước mặt (Flood-fill ratio)
        flood_ratio = self._calculate_flood_fill_ratio(pt_straight_1)

        state = [
            float(danger_s1),
            float(danger_r1),
            float(danger_l1),
            float(danger_s2),
            float(danger_r2),
            float(danger_l2),
            float(dir_up),
            float(dir_right_flag),
            float(dir_down),
            float(dir_left_flag),
            float(food_straight),
            float(food_is_right),
            float(food_is_left),
            float(food_behind),
            float(dx),
            float(dy),
            float(euclidean_dist),
            float(snake_len_ratio),
            float(ray_straight),
            float(ray_right),
            float(ray_left),
            float(flood_ratio),
        ]

        return np.array(state, dtype=np.float32)

    def step(self, action):
        """
        Thực hiện một bước đi:
        action:
            0 hoặc [1, 0, 0]: Đi thẳng
            1 hoặc [0, 1, 0]: Rẽ trái
            2 hoặc [0, 0, 1]: Rẽ phải
        Trả về: (next_state, reward, done, score)
        """
        self.steps += 1
        self.steps_since_last_food += 1

        # Chuyển đổi định dạng action nếu là int hoặc list/array
        if isinstance(action, (list, tuple, np.ndarray)):
            action_idx = int(np.argmax(action))
        else:
            action_idx = int(action)

        # Xác định hướng đi mới
        dir_idx = CLOCKWISE_DIRECTIONS.index(self.direction)
        if action_idx == 0:
            new_dir = CLOCKWISE_DIRECTIONS[dir_idx]  # Đi thẳng
        elif action_idx == 1:
            new_dir = CLOCKWISE_DIRECTIONS[(dir_idx - 1) % 4]  # Rẽ trái
        elif action_idx == 2:
            new_dir = CLOCKWISE_DIRECTIONS[(dir_idx + 1) % 4]  # Rẽ phải
        else:
            raise ValueError(f"Hành động không hợp lệ: {action_idx}")

        self.direction = new_dir

        # Tính khoảng cách tới mồi trước khi di chuyển
        prev_dist = 0
        if self.food is not None:
            prev_dist = abs(self.food[0] - self.head[0]) + abs(self.food[1] - self.head[1])

        # Tính toạ độ mới của đầu rắn
        new_head = (self.head[0] + self.direction.value[0], self.head[1] + self.direction.value[1])

        reward = 0.0
        done = False

        # Kiểm tra va chạm (chết do đâm vào tường hoặc tự cắn vào thân)
        if self.is_collision(new_head):
            self.head = new_head  # Cập nhật đầu rắn tới vị trí đâm để vẽ giao diện chết
            reward = -10.0
            done = True
            return self.get_state(), reward, done, self.score

        # Giới hạn số bước di chuyển không ăn mồi để tránh lặp vô tận (Starvation)
        max_idle_steps = 100 + len(self.snake) * 10
        if self.steps_since_last_food > max_idle_steps:
            reward = -10.0
            done = True
            return self.get_state(), reward, done, self.score

        # Di chuyển đầu rắn tới ô mới an toàn
        self.head = new_head
        self.snake.insert(0, self.head)
        self.snake_set.add(self.head)

        # Kiểm tra ăn mồi
        if self.food is not None and self.head == self.food:
            self.score += 1
            reward = 10.0
            self.steps_since_last_food = 0
            self._place_food()
            if self.food is None:
                # Đã thắng trò chơi (chiếm toàn bộ màn 14x14)
                reward += 50.0
                done = True
        else:
            # Không ăn mồi thì co đuôi lại
            tail = self.snake.pop()
            if tail not in self.snake:
                self.snake_set.remove(tail)

            # Thưởng/Phạt định hướng dựa trên việc lại gần hay rời xa mồi
            if self.food is not None:
                curr_dist = abs(self.food[0] - self.head[0]) + abs(self.food[1] - self.head[1])
                if curr_dist < prev_dist:
                    reward += 0.2  # Lại gần mồi
                else:
                    reward -= 0.2  # Rời xa mồi

            # Phạt nhẹ theo thời gian để khuyến khích đường đi ngắn nhất
            reward -= 0.01

        return self.get_state(), reward, done, self.score

    def render(self, fps=20, info_text=""):
        """Vẽ trò chơi lên màn hình Pygame"""
        if not self.render_mode:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Màu sắc hiện đại
        BG_COLOR = (24, 28, 36)
        GRID_LINE = (34, 40, 52)
        SNAKE_HEAD = (46, 204, 113)
        SNAKE_BODY = (39, 174, 96)
        SNAKE_EYE = (255, 255, 255)
        FOOD_COLOR = (231, 76, 60)
        PANEL_BG = (18, 22, 28)
        TEXT_COLOR = (236, 240, 241)

        self.display.fill(BG_COLOR)

        # Vẽ lưới 14x14
        for x in range(0, self.grid_size * self.cell_size + 1, self.cell_size):
            pygame.draw.line(self.display, GRID_LINE, (x, 0), (x, self.grid_size * self.cell_size))
        for y in range(0, self.grid_size * self.cell_size + 1, self.cell_size):
            pygame.draw.line(self.display, GRID_LINE, (0, y), (self.grid_size * self.cell_size, y))

        # Vẽ mồi ăn
        if self.food is not None:
            fx = self.food[0] * self.cell_size
            fy = self.food[1] * self.cell_size
            pygame.draw.rect(
                self.display,
                FOOD_COLOR,
                pygame.Rect(fx + 3, fy + 3, self.cell_size - 6, self.cell_size - 6),
                border_radius=8,
            )

        # Vẽ thân rắn
        for i, pt in enumerate(self.snake):
            sx = pt[0] * self.cell_size
            sy = pt[1] * self.cell_size
            is_head = i == 0
            color = SNAKE_HEAD if is_head else SNAKE_BODY
            rect = pygame.Rect(sx + 2, sy + 2, self.cell_size - 4, self.cell_size - 4)
            pygame.draw.rect(self.display, color, rect, border_radius=6 if is_head else 4)

            # Vẽ mắt cho đầu rắn
            if is_head:
                eye_radius = 3
                if self.direction == Direction.RIGHT:
                    e1 = (sx + self.cell_size - 8, sy + 8)
                    e2 = (sx + self.cell_size - 8, sy + self.cell_size - 8)
                elif self.direction == Direction.LEFT:
                    e1 = (sx + 8, sy + 8)
                    e2 = (sx + 8, sy + self.cell_size - 8)
                elif self.direction == Direction.UP:
                    e1 = (sx + 8, sy + 8)
                    e2 = (sx + self.cell_size - 8, sy + 8)
                else:  # DOWN
                    e1 = (sx + 8, sy + self.cell_size - 8)
                    e2 = (sx + self.cell_size - 8, sy + self.cell_size - 8)
                pygame.draw.circle(self.display, SNAKE_EYE, e1, eye_radius)
                pygame.draw.circle(self.display, SNAKE_EYE, e2, eye_radius)

        # Vẽ bảng thông tin bên dưới bản đồ
        panel_rect = pygame.Rect(0, self.grid_size * self.cell_size, self.width, 60)
        pygame.draw.rect(self.display, PANEL_BG, panel_rect)
        pygame.draw.line(self.display, (52, 73, 94), (0, self.grid_size * self.cell_size), (self.width, self.grid_size * self.cell_size), 2)

        score_surface = self.font.render(f"Điểm: {self.score}  |  Bước: {self.steps}", True, TEXT_COLOR)
        self.display.blit(score_surface, (15, self.grid_size * self.cell_size + 8))

        if info_text:
            info_surface = self.small_font.render(info_text, True, (189, 195, 199))
            self.display.blit(info_surface, (15, self.grid_size * self.cell_size + 34))

        pygame.display.flip()
        if self.clock is not None and fps > 0:
            self.clock.tick(fps)
