"""
Lớp SnakeGame: Quản lý logic trò chơi Rắn săn mồi trên lưới 14x14.
- 3 đến 4 vật cản cố định trên bản đồ
- Thân rắn là vật cản động di chuyển theo từng bước
- Rắn tương tác thông qua 3 toán tử chính: Đi thẳng, Rẽ trái, Rẽ phải
"""

import random
from constants import (
    Point, GRID_WIDTH, GRID_HEIGHT,
    CLOCKWISE_DIRS, DIR_RIGHT,
    ACTION_STRAIGHT, ACTION_LEFT, ACTION_RIGHT,
    DEFAULT_NUM_OBSTACLES
)
from obstacle_detector import ObstacleDetector

class SnakeGame:
    def __init__(self, num_obstacles: int = DEFAULT_NUM_OBSTACLES):
        self.grid_w = GRID_WIDTH
        self.grid_h = GRID_HEIGHT
        self.num_obstacles = num_obstacles
        self.record_score = 0
        self.reset()

    def reset(self, randomize_obstacles: bool = True):
        """Khởi tạo lại trạng thái trò chơi."""
        # 1. Hướng ban đầu: Sang phải
        self.direction = DIR_RIGHT

        # 2. Vị trí ban đầu của rắn ở gần tâm bản đồ 14x14
        start_x = self.grid_w // 2
        start_y = self.grid_h // 2
        self.head = Point(start_x, start_y)
        self.body = [
            Point(start_x - 1, start_y),
            Point(start_x - 2, start_y)
        ]

        # 3. Tạo 3 - 4 vật cản cố định (không trùng thân rắn và vùng khởi đầu)
        if randomize_obstacles or not hasattr(self, 'obstacles'):
            self.obstacles = self._generate_obstacles(self.num_obstacles)

        # 4. Sinh mồi ngẫu nhiên
        self.food = None
        self._place_food()

        # 5. Thông số trò chơi
        self.score = 0
        self.steps = 0
        self.is_game_over = False
        self.last_action = ACTION_STRAIGHT
        self.last_operator_name = "Đi thẳng"

    def _generate_obstacles(self, count: int) -> set:
        """
        Sinh ngẫu nhiên 3 đến 4 ô vật cản cố định trên bản đồ 14x14.
        Đảm bảo không chặn đầu rắn ban đầu và phân bố hợp lý.
        """
        obstacles = set()
        reserved_zone = {
            self.head,
            Point(self.head.x + 1, self.head.y),
            Point(self.head.x + 2, self.head.y),
            Point(self.head.x - 1, self.head.y),
            Point(self.head.x - 2, self.head.y),
            Point(self.head.x, self.head.y - 1),
            Point(self.head.x, self.head.y + 1),
        }
        for seg in self.body:
            reserved_zone.add(seg)

        attempts = 0
        while len(obstacles) < count and attempts < 200:
            attempts += 1
            ox = random.randint(1, self.grid_w - 2)
            oy = random.randint(1, self.grid_h - 2)
            pt = Point(ox, oy)
            if pt not in reserved_zone and pt not in obstacles:
                obstacles.add(pt)

        return obstacles

    def _place_food(self):
        """Đặt mồi ngẫu nhiên vào ô trống không bị chiếm dụng."""
        occupied = set(self.obstacles)
        occupied.add(self.head)
        for seg in self.body:
            occupied.add(seg)

        available_points = [
            Point(x, y)
            for x in range(self.grid_w)
            for y in range(self.grid_h)
            if Point(x, y) not in occupied
        ]

        if available_points:
            self.food = random.choice(available_points)
        else:
            # Thắng game nếu chiếm hết toàn bộ ô trống
            self.food = None

    def play_step(self, action: list) -> tuple:
        """
        Thực hiện một bước đi theo 1 trong 3 toán tử:
        action = [1, 0, 0] -> Đi thẳng (Straight)
        action = [0, 1, 0] -> Rẽ trái  (Turn Left)
        action = [0, 0, 1] -> Rẽ phải  (Turn Right)

        Trả về: (reward, game_over, score)
        """
        self.steps += 1
        self.last_action = action

        # 1. Xác định hướng di chuyển mới dựa trên 3 toán tử
        dir_idx = CLOCKWISE_DIRS.index(self.direction)
        if action == ACTION_STRAIGHT or action == [1, 0, 0]:
            self.direction = CLOCKWISE_DIRS[dir_idx]
            self.last_operator_name = "Đi thẳng"
        elif action == ACTION_LEFT or action == [0, 1, 0]:
            # Rẽ trái (ngược chiều kim đồng hồ)
            self.direction = CLOCKWISE_DIRS[(dir_idx - 1) % 4]
            self.last_operator_name = "Rẽ trái"
        elif action == ACTION_RIGHT or action == [0, 0, 1]:
            # Rẽ phải (theo chiều kim đồng hồ)
            self.direction = CLOCKWISE_DIRS[(dir_idx + 1) % 4]
            self.last_operator_name = "Rẽ phải"

        # 2. Tọa độ đầu rắn ở bước kế tiếp
        new_head = Point(self.head.x + self.direction.x, self.head.y + self.direction.y)

        # 3. Kiểm tra va chạm (Tường biên, Vật cản cố định, Thân rắn)
        # Chú ý: Đuôi rắn ở bước này sẽ di chuyển nếu không ăn mồi
        will_eat_food = (new_head == self.food)
        if ObstacleDetector.is_collision(new_head, self.body, self.obstacles,
                                        self.grid_w, self.grid_h,
                                        ignore_tail=not will_eat_food):
            self.is_game_over = True
            return -10, self.is_game_over, self.score

        # 4. Cập nhật vị trí thân rắn (Vật cản động di chuyển theo từng bước)
        self.body.insert(0, self.head)
        self.head = new_head

        # 5. Kiểm tra ăn mồi
        reward = 0
        if will_eat_food:
            self.score += 1
            reward = 10
            if self.score > self.record_score:
                self.record_score = self.score
            self._place_food()
        else:
            # Thân rắn di chuyển dời đi -> xóa đốt đuôi cũ
            self.body.pop()

        return reward, self.is_game_over, self.score

    def get_full_snake(self) -> list:
        """Trả về toàn bộ các điểm của rắn [head, seg_1, seg_2, ...]"""
        return [self.head] + list(self.body)
