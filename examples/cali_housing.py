import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from core.engine import Value
from nn.nn import MLP
from nn.losses import MSELoss
from optim.optim import AdamW  # Using AdamW for fun

housing = fetch_california_housing()
X, y = housing.data, housing.target

# Reshape y to be a column vector (N, 1) for broadcasting with predictions
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Input features: 8
# Hidden layers: Two hidden layers with 32 and 16 neurons respectively (with tanh activation).
# Output: 1 neuron for the predicted house value (no activation).
model = MLP(num_inputs=8, layer_sizes=[32, 16, 1])

optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)

criterion = MSELoss()

epochs = 200
batch_size = 256

for epoch in range(epochs):
    # Simple batching for demonstration
    permutation = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[permutation]
    y_train_shuffled = y_train[permutation]

    epoch_loss = 0
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train_shuffled[i:i + batch_size]
        y_batch = y_train_shuffled[i:i + batch_size]

        inputs = Value(X_batch)
        predictions = model(inputs)
        loss = criterion(predictions, y_batch)
        epoch_loss += loss.data
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 20 == 0 or epoch == epochs - 1:
        avg_epoch_loss = epoch_loss / (X_train.shape[0] / batch_size)

        test_inputs = Value(X_test)
        test_preds = model(test_inputs)
        test_loss = criterion(test_preds, y_test)

        # We can also compute R^2 score for a more interpretable metric
        y_test_mean = y_test.mean()
        ss_total = ((y_test - y_test_mean) ** 2).sum()
        ss_res = ((y_test - test_preds.data) ** 2).sum()
        r2 = 1 - (ss_res / ss_total)

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_epoch_loss:.4f}, Test MSE: {test_loss.data:.4f}, Test R^2: {r2:.4f}")

print("\n---------------------------------")
print(f"Model Training Complete.")
# Example prediction
sample_idx = 10
pred = model(Value(X_test[sample_idx:sample_idx + 1])).data[0, 0]
actual = y_test[sample_idx, 0]
print(f"Sample prediction: ${pred * 100_000:,.2f} (Actual: ${actual * 100_000:,.2f})")
print("---------------------------------")
