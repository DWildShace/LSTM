"""
Module phát hiện vật cản (Obstacle Detection Module) và cảm biến môi trường.
Hỗ trợ kiểm tra:
1. Tường biên bản đồ (14x14)
2. Vật cản cố định (3-4 vật cản)
3. Thân rắn (vật cản động di chuyển theo từng bước)
"""

from constants import (
    Point, GRID_WIDTH, GRID_HEIGHT,
    CLOCKWISE_DIRS, DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT
)

class ObstacleDetector:
    """
    Lớp cung cấp các phương thức cảm biến và phát hiện vật cản cho rắn.
    """

    @staticmethod
    def is_out_of_bounds(point: Point, grid_w: int = GRID_WIDTH, grid_h: int = GRID_HEIGHT) -> bool:
        """Kiểm tra xem điểm có va chạm với tường biên hay không."""
        return point.x < 0 or point.x >= grid_w or point.y < 0 or point.y >= grid_h

    @staticmethod
    def is_static_obstacle(point: Point, obstacles: set) -> bool:
        """Kiểm tra xem điểm có trùng với vật cản cố định hay không."""
        return point in obstacles

    @staticmethod
    def is_body_collision(point: Point, snake_body: list, ignore_tail: bool = False) -> bool:
        """
        Kiểm tra va chạm với thân rắn (vật cản động).
        Nếu ignore_tail=True, bỏ qua đốt đuôi cuối cùng (vì đuôi sẽ dời đi khi rắn bước tiếp).
        """
        if not snake_body:
            return False
        
        body_to_check = snake_body[:-1] if (ignore_tail and len(snake_body) > 1) else snake_body
        return point in body_to_check

    @classmethod
    def is_collision(cls, point: Point, snake_body: list, obstacles: set,
                     grid_w: int = GRID_WIDTH, grid_h: int = GRID_HEIGHT,
                     ignore_tail: bool = False) -> bool:
        """
        Kiểm tra toàn diện va chạm:
        - Tường biên (14x14)
        - Vật cản cố định (3-4 vật cản)
        - Thân rắn (vật cản động)
        """
        # 1. Tường biên
        if cls.is_out_of_bounds(point, grid_w, grid_h):
            return True
        # 2. Vật cản cố định
        if cls.is_static_obstacle(point, obstacles):
            return True
        # 3. Thân rắn (vật cản động)
        if cls.is_body_collision(point, snake_body, ignore_tail=ignore_tail):
            return True
        return False

    @classmethod
    def get_danger_sensors(cls, head: Point, current_dir: Point,
                           snake_body: list, obstacles: set,
                           grid_w: int = GRID_WIDTH, grid_h: int = GRID_HEIGHT) -> dict:
        """
        Cảm biến vật cản theo 3 hướng tương đối của đầu rắn:
        - Thẳng (Straight)
        - Trái (Left)
        - Phải (Right)
        
        Trả về dict { 'straight': bool, 'left': bool, 'right': bool }
        """
        dir_idx = CLOCKWISE_DIRS.index(current_dir)
        dir_straight = CLOCKWISE_DIRS[dir_idx]
        dir_right = CLOCKWISE_DIRS[(dir_idx + 1) % 4]
        dir_left = CLOCKWISE_DIRS[(dir_idx - 1) % 4]

        pt_straight = Point(head.x + dir_straight.x, head.y + dir_straight.y)
        pt_right = Point(head.x + dir_right.x, head.y + dir_right.y)
        pt_left = Point(head.x + dir_left.x, head.y + dir_left.y)

        danger_straight = cls.is_collision(pt_straight, snake_body, obstacles, grid_w, grid_h)
        danger_right = cls.is_collision(pt_right, snake_body, obstacles, grid_w, grid_h)
        danger_left = cls.is_collision(pt_left, snake_body, obstacles, grid_w, grid_h)

        return {
            'straight': danger_straight,
            'left': danger_left,
            'right': danger_right,
            'pt_straight': pt_straight,
            'pt_left': pt_left,
            'pt_right': pt_right
        }

    @classmethod
    def get_sensor_state(cls, head: Point, current_dir: Point,
                         snake_body: list, obstacles: set, food: Point,
                         grid_w: int = GRID_WIDTH, grid_h: int = GRID_HEIGHT) -> list:
        """
        Vector trạng thái chuẩn AI (State representation):
        [
            danger_straight, danger_left, danger_right,
            dir_up, dir_right, dir_down, dir_left,
            food_up, food_right, food_down, food_left
        ]
        """
        dangers = cls.get_danger_sensors(head, current_dir, snake_body, obstacles, grid_w, grid_h)

        state = [
            # 3 cảm biến vật cản
            int(dangers['straight']),
            int(dangers['left']),
            int(dangers['right']),

            # Hướng di chuyển hiện tại
            int(current_dir == DIR_UP),
            int(current_dir == DIR_RIGHT),
            int(current_dir == DIR_DOWN),
            int(current_dir == DIR_LEFT),

            # Hướng mồi tương đối
            int(food.y < head.y), # Mồi ở phía trên
            int(food.x > head.x), # Mồi ở phía phải
            int(food.y > head.y), # Mồi ở phía dưới
            int(food.x < head.x), # Mồi ở phía trái
        ]
        return state

    @classmethod
    def get_grid_matrix(cls, head: Point, snake_body: list, obstacles: set,
                        food: Point, grid_w: int = GRID_WIDTH, grid_h: int = GRID_HEIGHT) -> list:
        """
        Xuất ma trận lưới 14x14 mô tả môi trường:
        0: Ô trống
        1: Vật cản cố định (Static Obstacle)
        2: Thân rắn (Dynamic Obstacle)
        3: Đầu rắn (Snake Head)
        4: Mồi (Food)
        """
        matrix = [[0 for _ in range(grid_w)] for _ in range(grid_h)]

        for obs in obstacles:
            if 0 <= obs.x < grid_w and 0 <= obs.y < grid_h:
                matrix[obs.y][obs.x] = 1

        for segment in snake_body:
            if 0 <= segment.x < grid_w and 0 <= segment.y < grid_h:
                matrix[segment.y][segment.x] = 2

        if 0 <= head.x < grid_w and 0 <= head.y < grid_h:
            matrix[head.y][head.x] = 3

        if food and 0 <= food.x < grid_w and 0 <= food.y < grid_h:
            matrix[food.y][food.x] = 4

        return matrix
