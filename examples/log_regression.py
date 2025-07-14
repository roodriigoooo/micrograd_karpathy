import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression as SkLogReg
from sklearn.metrics import accuracy_score
from core.engine import Value


class LogisticRegression:
    """A plain logistic‑regression model implemented with milligrad."""

    def __init__(self, n_features: int):
        # Xavier/Glorot uniform initialisation
        limit = np.sqrt(6 / n_features)
        self.W = Value(np.random.uniform(-limit, limit, (n_features, 1)))  # (D,1)
        self.b = Value(np.zeros((1,)))

    def __call__(self, x: Value) -> Value:
        """Forward pass producing *probabilities* in (0,1)."""
        logits = x @ self.W + self.b  # (N,1)
        # σ(z) = 1 / (1 + e^(−z))
        probs = ((-logits).exp() + 1) ** -1
        return probs

    # convenience helpers -------------------------------------------------
    def parameters(self):
        return [self.W, self.b]

    def zero_grad(self):
        for p in self.parameters():
            p.grad.fill(0)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def binary_cross_entropy(pred: Value, target: Value) -> Value:
    """Mean binary cross‑entropy over a batch (numerically stable)."""
    eps = 1e-7  # to avoid log(0)
    loss = -(target * (pred + eps).log() + (1 - target) * (1 - pred + eps).log()).sum() / target.data.shape[0]
    return loss


def train():
    X, y = make_moons(n_samples=500, noise=0.25, random_state=42)
    X_val, y_val = make_moons(n_samples=200, noise=0.25, random_state=0)

    model = LogisticRegression(n_features=2)

    lr = 0.1
    epochs = 1000

    for epoch in range(epochs):
        inputs = Value(X.astype(np.float32))               # (N,2)
        targets = Value(y.reshape(-1, 1).astype(np.float32))  # (N,1)

        preds = model(inputs)
        loss = binary_cross_entropy(preds, targets)

        # Back‑prop
        model.zero_grad()
        loss.backward()

        # SGD update
        for p in model.parameters():
            p.data -= lr * p.grad

        if epoch % 100 == 0:
            print(f"epoch {epoch:04d} | loss {loss.data:.4f}")

    X_val_v = Value(X_val.astype(np.float32))
    y_val_v = y_val.reshape(-1, 1).astype(np.float32)

    val_preds = (model(X_val_v).data > 0.5).astype(int)
    acc = accuracy_score(y_val_v, val_preds) * 100
    print(f"\nValidation accuracy (milligrad): {acc:.2f}%")

    # scikit‑learn baseline ----------------------------------------------
    skl = SkLogReg().fit(X, y)
    skl_acc = skl.score(X_val, y_val) * 100
    print(f"Validation accuracy (scikit‑learn): {skl_acc:.2f}%")

    make_decision_boundary_plot(model, X, y)


def make_decision_boundary_plot(model: LogisticRegression, X: np.ndarray, y: np.ndarray):
    """render the 2‑D decision surface produced by *our* model."""
    h = 0.02  # mesh‑step
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    probs = model(Value(grid)).data.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.25)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k", linewidth=0.5, cmap="coolwarm")
    plt.title("Decision boundary – milligrad logistic regression")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train()
