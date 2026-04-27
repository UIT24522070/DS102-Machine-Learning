import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


class SVM:
    """
    Soft-margin SVM implemented from scratch using NumPy.
    Trained with Stochastic Gradient Descent (SGD).
    """

    # ==========================================================
    # BLOCK 1: KHỞI TẠO
    # ==========================================================
    def __init__(self, C: float = 1.0, lr: float = 0.001, n_iterations: int = 50):
        self.C            = C              # regularization
        self.lr           = lr             # learning rate
        self.n_iterations = n_iterations   # số epoch
        self.losses       = []             # lưu loss mỗi epoch
        self.W            = None           # weight vector
        self.b            = 0.0            # bias

    # ==========================================================
    # BLOCK 2: TRAINING
    # ==========================================================
    def fit(self, X: np.ndarray, y: np.ndarray):
        N, dim = X.shape
        self.W = np.zeros(dim, dtype=np.float64)
        self.b = 0.0

        for epoch in range(self.n_iterations):
            # Learning rate decay: càng về sau bước nhảy càng nhỏ
            lr_t = self.lr / (1 + 0.01 * epoch)

            # Shuffle mỗi epoch: tránh học theo thứ tự ảnh
            idx  = np.random.permutation(N)
            X, y = X[idx], y[idx]

            epoch_loss = 0.0
            for i in range(N):
                xi  = X[i]
                yi  = y[i]

                # Raw score: chưa sign(), dùng để tính gradient
                raw = float(np.dot(xi, self.W) + self.b)

                # Gradient của Hinge loss + Regularization
                if yi * raw >= 1:
                    # Đúng + ngoài margin → chỉ có regularization
                    dW = self.W
                    db = 0.0
                else:
                    # Sai hoặc trong margin → có hinge loss
                    dW = self.W - self.C * yi * xi
                    db = -self.C * yi

                # Update W và b
                self.W -= lr_t * dW
                self.b -= lr_t * db

                # Tính loss sau update để theo dõi
                raw_new    = float(np.dot(xi, self.W) + self.b)
                hinge      = max(0.0, 1 - yi * raw_new)
                epoch_loss += 0.5 * float(np.dot(self.W, self.W)) + self.C * hinge

            avg_loss = epoch_loss / N
            self.losses.append(avg_loss)
            print(f"Epoch {epoch+1:>3}/{self.n_iterations} | Loss: {avg_loss:.6f}")

    # ==========================================================
    # BLOCK 3: PREDICT
    # ==========================================================
    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = np.dot(X, self.W) + self.b
        return np.sign(raw).astype(int)

    # ==========================================================
    # BLOCK 4: EVALUATE
    # ==========================================================
    def get_metrics(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        y_pred = np.where(y_pred >= 0, 1, -1)

        P  = precision_score(y, y_pred, zero_division=0)
        R  = recall_score   (y, y_pred, zero_division=0)
        F1 = f1_score       (y, y_pred, zero_division=0)

        return {"Precision": P, "Recall": R, "F1": F1}