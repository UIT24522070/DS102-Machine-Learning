import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from SVM import SVM

BASE_DIR = "/Applications/DaiHoc/DS102-Machine-Learning/Lab3/chest_xray"

# ==========================================================
# BLOCK 1: LOAD DATA
# ==========================================================
def collect_data(split: str = "train", img_size: int = 128):
    data = {
        "NORMAL"   :  1,   # NORMAL    = +1
        "PNEUMONIA": -1    # PNEUMONIA = -1
    }
    images, labels = [], []

    for folder, label in data.items():
        folder_path = os.path.join(BASE_DIR, split, folder)
        files       = os.listdir(folder_path)

        for fname in tqdm(files, desc=f"{split}/{folder}"):
            path = os.path.join(folder_path, fname)
            img  = cv.imread(path)
            if img is None:
                continue
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.resize(img, (img_size, img_size), interpolation=cv.INTER_LINEAR)
            images.append(img)
            labels.append(label)

    X = np.stack(images, axis=0)
    y = np.array(labels)
    return X, y


# ==========================================================
# BLOCK 2: PREPROCESS
# ==========================================================
def preprocess(X_train, X_test):
    # Flatten: (N, 128, 128) → (N, 16384)
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float64) / 255.0
    X_test  = X_test.reshape (X_test.shape[0],  -1).astype(np.float64) / 255.0

    # Zero-mean normalization theo train set
    mean = X_train.mean(axis=0)
    std  = X_train.std (axis=0) + 1e-8

    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std   # dùng mean/std của train

    return X_train, X_test


# ==========================================================
# BLOCK 3: PLOT LOSS
# ==========================================================
def plot_loss(losses):
    plt.figure(figsize=(10, 4))
    plt.plot(losses, color='steelblue', linewidth=1.5)
    plt.title("Training Loss per Epoch - SVM (NumPy SGD)")
    plt.xlabel("Epoch")
    plt.ylabel("Avg Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.show()


# ==========================================================
# BLOCK 4: MAIN
# ==========================================================
if __name__ == "__main__":
    # Load data
    print("=== Loading data ===")
    X_train_raw, y_train = collect_data("train")
    X_test_raw,  y_test  = collect_data("test")
    print(f"Train: {X_train_raw.shape} | Test: {X_test_raw.shape}")

    # Preprocess
    print("\n=== Preprocessing ===")
    X_train, X_test = preprocess(X_train_raw, X_test_raw)

    # Train
    print("\n=== Training ===")
    model = SVM(C=1.0, lr=0.001, n_iterations=50)
    model.fit(X_train, y_train)

    # Plot loss
    print("\n=== Loss Curve ===")
    plot_loss(model.losses)

    # Evaluate
    print("\n=== Evaluation ===")
    metrics = model.get_metrics(X_test, y_test)
    print(f"Precision : {metrics['Precision']:.4f}")
    print(f"Recall    : {metrics['Recall']:.4f}")
    print(f"F1 Score  : {metrics['F1']:.4f}")