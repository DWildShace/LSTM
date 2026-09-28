"""
File thực thi chính của dự án AI LSTM: Khởi động game Rắn săn mồi AI.
Cho phép chạy: py -3 main.py
"""

from play_ai import SnakeAILSTMVisualizer

if __name__ == "__main__":
    visualizer = SnakeAILSTMVisualizer()
    visualizer.run()
