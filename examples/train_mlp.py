import numpy as np
from core.engine import Value
from nn.nn import MLP
from optim.optim import SGD


def main():
    """Main training function."""
    # A simple binary classification problem
    xs_data = np.array([
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ])
    ys_data = np.array([[1.0], [-1.0], [-1.0], [1.0]])

    # Wrap data in Value objects
    xs = Value(xs_data)
    ys = Value(ys_data)

    # 2. Initialize the model and optimizer
    model = MLP(num_inputs=3, layer_sizes=[4, 4, 1])
    optimizer = SGD(model.parameters(), lr=0.05)

    print(f"\nModel created: {model}")
    print(f"Number of parameters: {len(model.parameters())}")

    # 3. Training loop
    epochs = 20
    print("\nStarting training...")
    for k in range(epochs):
        # Forward pass
        ypred = model(xs)
        loss = ((ypred - ys) ** 2).sum()
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()  # Compute new gradients
        optimizer.step()  # Update parameters

        if k % 5 == 0 or k == epochs - 1:
            print(f"Epoch {k + 1}/{epochs}, Loss: {loss.data:.4f}")

    # 4. Final predictions
    print("\nTraining finished.")
    final_preds = model(xs)
    print("\nFinal Predictions vs True Values:")
    for pred, true in zip(final_preds.data, ys.data):
        print(f"Pred: {pred[0]:.3f} | True: {true[0]}")


if __name__ == "__main__":
    main()