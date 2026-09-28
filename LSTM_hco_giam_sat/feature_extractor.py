"""
Trích xuất vector đặc trưng (Feature Extraction) từ môi trường game rắn săn mồi 14x14.
Kết hợp cảm biến va chạm từ ObstacleDetector và thông tin không gian tương đối.
"""

import sys
import os
import numpy as np

# Thiết lập đường dẫn thư mục để import an toàn
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
ASTAR_DIR = os.path.join(PARENT_DIR, "tim_kiem_A_sao")

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if ASTAR_DIR not in sys.path:
    sys.path.insert(0, ASTAR_DIR)

from constants import (  # noqa: E402
    Point, GRID_WIDTH, GRID_HEIGHT,
    CLOCKWISE_DIRS, DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT
)
from obstacle_detector import ObstacleDetector  # noqa: E402


def get_raycast_distance(head: Point, direction: Point, snake_body: list, 
                         obstacles: set, max_distance: int = 14) -> float:
    """
    Tính khoảng cách (số ô an toàn) từ đầu rắn theo 1 hướng nhất định
    trước khi chạm vào tường, chướng ngại vật cố định hoặc thân rắn.
    Chuẩn hóa về đoạn [0.0, 1.0].
    """
    cur_x, cur_y = head.x, head.y
    distance = 0
    for step in range(1, max_distance + 1):
        test_pt = Point(cur_x + direction.x * step, cur_y + direction.y * step)
        if ObstacleDetector.is_collision(test_pt, snake_body, obstacles, GRID_WIDTH, GRID_HEIGHT):
            break
        distance = step
    return distance / float(max_distance)


def extract_features(head: Point, current_dir: Point, snake_body: list, 
                     obstacles: set, food: Point) -> np.ndarray:
    """
    Trích xuất vector 16 đặc trưng tại 1 thời điểm t:
    - 3 cảm biến nguy hiểm liền kề (Thẳng, Trái, Phải): 0 hoặc 1
    - 4 hướng di chuyển hiện tại (Lên, Phải, Xuống, Trái): 0 hoặc 1
    - 4 vị trí tương đối của mồi (Trên, Phải, Dưới, Trái): 0 hoặc 1
    - 2 khoảng cách tọa độ chuẩn hóa đến mồi (dx, dy): [-1.0, 1.0]
    - 3 khoảng cách raycast nhìn xa theo 3 hướng tương đối (Thẳng, Trái, Phải): [0.0, 1.0]
    """
    # 1. Ba hướng tương đối theo đầu rắn
    dir_idx = CLOCKWISE_DIRS.index(current_dir)
    dir_straight = CLOCKWISE_DIRS[dir_idx]
    dir_left = CLOCKWISE_DIRS[(dir_idx - 1) % 4]
    dir_right = CLOCKWISE_DIRS[(dir_idx + 1) % 4]

    # 2. Cảm biến va chạm tức thời 1 bước
    dangers = ObstacleDetector.get_danger_sensors(
        head, current_dir, snake_body, obstacles, GRID_WIDTH, GRID_HEIGHT
    )

    # 3. Hướng mồi & Khoảng cách mồi
    if food is not None:
        food_up = float(food.y < head.y)
        food_right = float(food.x > head.x)
        food_down = float(food.y > head.y)
        food_left = float(food.x < head.x)
        norm_dx = (food.x - head.x) / float(GRID_WIDTH)
        norm_dy = (food.y - head.y) / float(GRID_HEIGHT)
    else:
        food_up = food_right = food_down = food_left = 0.0
        norm_dx = norm_dy = 0.0

    # 4. Khoảng cách raycast nhìn xa (Tầm nhìn hành lang an toàn)
    dist_straight = get_raycast_distance(head, dir_straight, snake_body, obstacles, GRID_WIDTH)
    dist_left = get_raycast_distance(head, dir_left, snake_body, obstacles, GRID_WIDTH)
    dist_right = get_raycast_distance(head, dir_right, snake_body, obstacles, GRID_WIDTH)

    features = [
        # [0, 1, 2]: Nguy hiểm ngay trước mắt (1 bước)
        float(dangers['straight']),
        float(dangers['left']),
        float(dangers['right']),

        # [3, 4, 5, 6]: Hướng di chuyển hiện tại
        float(current_dir == DIR_UP),
        float(current_dir == DIR_RIGHT),
        float(current_dir == DIR_DOWN),
        float(current_dir == DIR_LEFT),

        # [7, 8, 9, 10]: Vị trí tương đối của mồi
        food_up,
        food_right,
        food_down,
        food_left,

        # [11, 12]: Khoảng cách chuẩn hóa tới mồi
        norm_dx,
        norm_dy,

        # [13, 14, 15]: Tầm nhìn hành lang (Khoảng cách an toàn tới vật cản)
        dist_straight,
        dist_left,
        dist_right
    ]

    return np.array(features, dtype=np.float32)
