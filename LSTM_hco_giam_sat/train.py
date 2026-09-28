"""
Huấn luyện mô hình AI LSTM chơi Rắn săn mồi từ dữ liệu nhãn A*.
Sử dụng Keras (hỗ trợ cả backend PyTorch / TensorFlow).
Tự động áp dụng Class Weighting để cân bằng xác suất rẽ trái/phải/thẳng.
"""

import sys
import os
import numpy as np

# Đảm bảo in tiếng Việt chuẩn trên Windows console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Đặt Keras backend sang torch mặc định nếu chưa set
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
ASTAR_DIR = os.path.join(PARENT_DIR, "tim_kiem_A_sao")

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if ASTAR_DIR not in sys.path:
    sys.path.insert(0, ASTAR_DIR)

from config import (  # noqa: E402
    DATASET_PATH, MODEL_PATH, MODEL_H5_PATH,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, TEST_SPLIT, SEQ_LEN, FEATURE_DIM
)
from model import build_keras_model  # noqa: E402
from collect_data import collect_expert_demonstrations  # noqa: E402


import argparse  # noqa: E402


def train(epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, lr: float = LEARNING_RATE, resume: bool = False):
    # 1. Kiểm tra tập dữ liệu
    if not os.path.exists(DATASET_PATH):
        print(f"[!] Không tìm thấy tập dữ liệu tại: {DATASET_PATH}")
        print("[*] Đang tự động gọi thu thập dữ liệu nhãn từ thuật toán A*...")
        collect_expert_demonstrations(num_games=150)

    print(f"[*] Đang tải dữ liệu từ: {DATASET_PATH}")
    data = np.load(DATASET_PATH)
    X = data['X']  # (N, seq_len, feature_dim)
    Y = data['Y']  # (N, 3)

    data_seq_len = X.shape[1]
    if data_seq_len != SEQ_LEN:
        print("\n" + "!" * 70)
        print(f"[!] PHÁT HIỆN LỆCH ĐỘ DÀI CHUỖI (SEQ_LEN):")
        print(f"    - Tập dữ liệu cũ đang có seq_len = {data_seq_len}")
        print(f"    - Nhưng cấu hình config.py hiện tại là SEQ_LEN = {SEQ_LEN}")
        print(f"[*] Đang tự động tạo lại tập dữ liệu mới tương thích với SEQ_LEN = {SEQ_LEN}...")
        print("!" * 70 + "\n")
        collect_expert_demonstrations(num_games=NUM_COLLECT_GAMES, seq_len=SEQ_LEN)
        data = np.load(DATASET_PATH)
        X = data['X']
        Y = data['Y']

    num_samples = len(X)
    print(f" - Tổng số mẫu: {num_samples}")
    print(f" - Kích thước X: {X.shape}")
    print(f" - Kích thước Y: {Y.shape}")

    # 2. Xáo trộn ngẫu nhiên dữ liệu và chia Train/Validation
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    split_idx = int(num_samples * (1 - TEST_SPLIT))
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    print(f" - Tập huấn luyện (Train): {len(X_train)} mẫu")
    print(f" - Tập kiểm tra (Val)    : {len(X_val)} mẫu")

    # 3. Tính toán trọng số lớp (Class Weights) để giải quyết mất cân bằng nhãn
    y_cls = np.argmax(Y_train, axis=1)
    counts = np.bincount(y_cls, minlength=3)
    total = len(y_cls)
    class_weights = {i: float(total / (3.0 * max(counts[i], 1))) for i in range(3)}
    print(f" - Trọng số cân bằng lớp (Class Weights): {class_weights}")

    # 4. Khởi tạo hoặc Tải mô hình để huấn luyện nối tiếp (Resume)
    import keras
    model = None
    if resume:
        target_load = MODEL_PATH if os.path.exists(MODEL_PATH) else (MODEL_H5_PATH if os.path.exists(MODEL_H5_PATH) else None)
        if target_load:
            try:
                print("\n" + "=" * 70)
                print(f"[*] TIẾP TỤC HUẤN LUYỆN NỐI TIẾP (RESUME) TỪ: {target_load}")
                loaded_model = keras.models.load_model(target_load)
                if loaded_model.input_shape[1] == SEQ_LEN:
                    model = loaded_model
                    print("[✓] Đã nạp thành công ma trận trọng số cũ!")
                else:
                    print(f"[!] Mô hình cũ có seq_len = {loaded_model.input_shape[1]}, không khớp SEQ_LEN = {SEQ_LEN} hiện tại.")
                    print(f"[*] Sẽ khởi tạo mô hình mới phù hợp với SEQ_LEN = {SEQ_LEN}!")
                    model = None
            except Exception as e:
                print(f"[!] Không thể nạp mô hình cũ ({e}). Khởi tạo mô hình mới!")
                model = None

    if model is None:
        print("\n" + "=" * 70)
        print(" KHỞI TẠO MÔ HÌNH MỚI (TÍCH HỢP LAYER NORMALIZATION):")
        model = build_keras_model(seq_len=SEQ_LEN, feature_dim=FEATURE_DIM, num_actions=3, use_layer_norm=True)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.02),
        metrics=["accuracy"]
    )
    model.summary()
    print("=" * 70)

    # 5. Thiết lập Callbacks tối ưu hóa loss
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1
        )
    ]

    # 6. Bắt đầu huấn luyện
    print(f"\n[*] Bắt đầu huấn luyện với {epochs} Epochs, Batch size = {batch_size}, LR = {lr}...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # 7. Đánh giá chất lượng mô hình
    val_loss, val_acc = model.evaluate(X_val, Y_val, verbose=0)
    print("\n" + "=" * 70)
    print(" KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (VALIDATION):")
    print(f" - Validation Loss    : {val_loss:.4f}")
    print(f" - Validation Accuracy: {val_acc * 100:.2f}%")
    print("=" * 70)

    # 8. Lưu mô hình đã huấn luyện
    try:
        model.save(MODEL_PATH)
        print(f"[✓] Đã lưu mô hình Keras v3 vào: {MODEL_PATH}")
    except Exception as e:
        print(f"[!] Không thể lưu định dạng .keras ({e}), đang lưu sang .h5...")
        model.save(MODEL_H5_PATH)
        print(f"[✓] Đã lưu mô hình H5 vào: {MODEL_H5_PATH}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình AI LSTM chơi Rắn săn mồi")
    parser.add_argument("-e", "--epochs", type=int, default=EPOCHS,
                        help=f"Số epoch huấn luyện (Mặc định: {EPOCHS})")
    parser.add_argument("-b", "--batch-size", "--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Kích thước mini-batch (Mặc định: {BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help=f"Tốc độ học Learning Rate (Mặc định: {LEARNING_RATE})")
    parser.add_argument("-r", "--resume", action="store_true",
                        help="Tiếp tục huấn luyện từ ma trận trọng số mô hình đã lưu trước đó")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, resume=args.resume)
