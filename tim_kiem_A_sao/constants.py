"""
Các hằng số và cấu hình cho trò chơi Rắn săn mồi A* (14x14).
"""

from collections import namedtuple

# Kích thước lưới bản đồ
GRID_WIDTH = 14
GRID_HEIGHT = 14
CELL_SIZE = 42  # Kích thước mỗi ô (pixel)
BOARD_WIDTH = GRID_WIDTH * CELL_SIZE   # 14 * 42 = 588 px
BOARD_HEIGHT = GRID_HEIGHT * CELL_SIZE # 14 * 42 = 588 px

# Kích thước cửa sổ tổng thể (Bản đồ + Bảng điều khiển Dashboard AI bên phải)
SIDEBAR_WIDTH = 350
WINDOW_WIDTH = BOARD_WIDTH + SIDEBAR_WIDTH
WINDOW_HEIGHT = BOARD_HEIGHT

# Điểm Point (Tọa độ ô trong bản đồ 14x14)
Point = namedtuple('Point', 'x, y')

# Các hướng di chuyển (theo thứ tự kim đồng hồ: Lên, Phải, Xuống, Trái)
DIR_UP = Point(0, -1)
DIR_RIGHT = Point(1, 0)
DIR_DOWN = Point(0, 1)
DIR_LEFT = Point(-1, 0)

CLOCKWISE_DIRS = [DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT]

# 3 Toán tử chính tương đối theo hướng đầu rắn:
# Index 0: Đi thẳng [1, 0, 0]
# Index 1: Rẽ trái  [0, 1, 0]
# Index 2: Rẽ phải  [0, 0, 1]
ACTION_STRAIGHT = [1, 0, 0]
ACTION_LEFT = [0, 1, 0]
ACTION_RIGHT = [0, 0, 1]

# Số lượng vật cản cố định (3 đến 4 vật cản)
DEFAULT_NUM_OBSTACLES = 4

# Bảng màu thẩm mỹ hiện đại (Sleek Dark Modern Palette)
COLOR_BG_DARK = (15, 23, 42)          # Màu nền slate dark #0f172a
COLOR_BOARD_BG = (17, 24, 39)         # Nền bàn cờ #111827
COLOR_GRID_LINE = (30, 41, 59)        # Viền ô lưới #1e293b
COLOR_SIDEBAR_BG = (23, 30, 48)       # Nền sidebar #171e30
COLOR_PANEL_BG = (30, 41, 59)         # Nền card thống kê #1e293b

# Màu các đối tượng trong game
COLOR_SNAKE_HEAD = (16, 185, 129)     # Đầu rắn xanh lục ngọc tươi #10b981
COLOR_SNAKE_BODY_START = (52, 211, 153) # Gradient thân rắn
COLOR_SNAKE_BODY_END = (5, 150, 105)
COLOR_SNAKE_EYES = (255, 255, 255)    # Mắt rắn
COLOR_SNAKE_PUPIL = (15, 23, 42)

COLOR_FOOD = (239, 68, 68)            # Mồi đỏ rực #ef4444
COLOR_FOOD_GLOW = (248, 113, 113)

COLOR_OBSTACLE = (245, 158, 11)       # Vật cản màu vàng cam cảnh báo #f59e0b
COLOR_OBSTACLE_BORDER = (217, 119, 6) # Viền vật cản
COLOR_OBSTACLE_STRIPE = (180, 83, 9)

# Màu hiển thị thuật toán A*
COLOR_ASTAR_PATH = (56, 189, 248)     # Đường đi A* xanh dương neon #38bdf8
COLOR_ASTAR_OPEN = (168, 85, 247)     # Tập mở node (tùy chọn)

# Màu text & UI
COLOR_TEXT_WHITE = (248, 250, 252)
COLOR_TEXT_MUTED = (148, 163, 184)
COLOR_TEXT_ACCENT = (56, 189, 248)
COLOR_DANGER = (239, 68, 68)          # Đỏ cảnh báo nguy hiểm
COLOR_SAFE = (34, 197, 94)            # Xanh lá an toàn

# Tốc độ khung hình mặc định
DEFAULT_FPS = 12
FAST_FPS = 30
SLOW_FPS = 5
