"""
File thực thi chính của dự án AI DRQN (Học Tăng Cường): Khởi động game Rắn săn mồi AI.
Cho phép chạy: py -3 main.py
"""

from play_ai import SnakeAIDRQNVisualizer

if __name__ == "__main__":
    visualizer = SnakeAIDRQNVisualizer()
    visualizer.run()
