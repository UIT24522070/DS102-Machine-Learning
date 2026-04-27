import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# Import từ Lab3.py để dùng lại hàm load + preprocess
from Lab3 import collect_data, preprocess


# ==========================================================
# BLOCK 1: TRAIN SKLEARN SVM
# ==========================================================
def train_sklearn_svm(X_train, y_train, n_epochs: int = 50):
    """
    Train SVM dùng SGDClassifier với hinge loss.
    Về bản chất giống LinearSVC nhưng train từng epoch
    nên có thể hiển thị progress bar.
    """
    model = SGDClassifier(
        loss         = "hinge",   # hinge loss = SVM
        alpha        = 0.0001,    # regularization
        max_iter     = 1,         # train từng epoch 1 lần
        warm_start   = True,      # giữ weights giữa các lần fit
        random_state = 42
    )

    for epoch in tqdm(range(n_epochs), desc="Training"):
        model.fit(X_train, y_train)

    return model


# ==========================================================
# BLOCK 2: EVALUATE
# ==========================================================
def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)

    P  = precision_score(y_test, y_pred, zero_division=0)
    R  = recall_score   (y_test, y_pred, zero_division=0)
    F1 = f1_score       (y_test, y_pred, zero_division=0)

    return {"Precision": P, "Recall": R, "F1": F1}


# ==========================================================
# BLOCK 3: SO SÁNH KẾT QUẢ
# ==========================================================
def plot_comparison(metrics_numpy: dict, metrics_sklearn: dict):
    metrics      = ["Precision", "Recall", "F1"]
    numpy_vals   = [metrics_numpy  [m] for m in metrics]
    sklearn_vals = [metrics_sklearn[m] for m in metrics]

    x     = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, numpy_vals,   width, label="SVM (NumPy SGD)", color="steelblue")
    bars2 = ax.bar(x + width/2, sklearn_vals, width, label="SVM (sklearn)",   color="coral")

    # Ghi số lên mỗi bar
    for bar in bars1 + bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.4f}",
            ha="center", va="bottom", fontsize=9
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.set_title("SVM: NumPy SGD vs sklearn")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150)
    plt.show()


# ==========================================================
# BLOCK 4: MAIN
# ==========================================================
if __name__ == "__main__":
    # Load + Preprocess
    print("=== Loading data ===")
    X_train_raw, y_train = collect_data("train")
    X_test_raw,  y_test  = collect_data("test")

    print("\n=== Preprocessing ===")
    X_train, X_test = preprocess(X_train_raw, X_test_raw)

    # Train sklearn SVM
    print("\n=== Training sklearn SVM ===")
    sklearn_model = train_sklearn_svm(X_train, y_train, n_epochs=50)

    # Evaluate sklearn
    print("\n=== Evaluation sklearn SVM ===")
    metrics_sklearn = evaluate(sklearn_model, X_test, y_test)
    print(f"Precision : {metrics_sklearn['Precision']:.4f}")
    print(f"Recall    : {metrics_sklearn['Recall']:.4f}")
    print(f"F1 Score  : {metrics_sklearn['F1']:.4f}")

    # Kết quả NumPy SVM từ Assignment 1
    metrics_numpy = {"Precision": 0.8544, "Recall": 0.3761, "F1": 0.5223}

    # So sánh
    print("\n=== So sánh kết quả ===")
    print("=" * 52)
    print(f"{'Model':<20} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("=" * 52)
    print(f"{'SVM (NumPy SGD)':<20} {metrics_numpy  ['Precision']:>10.4f} {metrics_numpy  ['Recall']:>8.4f} {metrics_numpy  ['F1']:>8.4f}")
    print(f"{'SVM (sklearn)'  :<20} {metrics_sklearn['Precision']:>10.4f} {metrics_sklearn['Recall']:>8.4f} {metrics_sklearn['F1']:>8.4f}")
    print("=" * 52)

    # Plot
    print("\n=== Comparison Chart ===")
    plot_comparison(metrics_numpy, metrics_sklearn)