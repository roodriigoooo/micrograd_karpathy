import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from core.engine import Value
from nn.nn import Layer
from nn.losses import CrossEntropyLoss
from optim.optim import Adam

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

# Scale features for better convergence
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = Layer(num_inputs=4, num_outputs=3, activation=None)

optimizer = Adam(model.parameters(), lr=0.1, weight_decay=0.001)

criterion = CrossEntropyLoss()


epochs = 100
for epoch in range(epochs):
    # Convert training data to milligrad Value object
    inputs = Value(X_train)
    logits = model(inputs)
    loss = criterion(logits, y_train)
    optimizer.zero_grad()  # Use optimizer's zero_grad for convenience
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0 or epoch == epochs - 1:
        test_inputs = Value(X_test)
        test_logits = model(test_inputs)

        # Get predictions by finding the index of the max logit
        predictions = np.argmax(test_logits.data, axis=1)
        acc = accuracy_score(y_test, predictions)
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {loss.data:.4f}, Test Accuracy: {acc:.4f}")


final_test_inputs = Value(X_test)
final_logits = model(final_test_inputs)
final_predictions = np.argmax(final_logits.data, axis=1)
final_accuracy = accuracy_score(y_test, final_predictions)

print("\n---------------------------------")
print(f"Final Test Accuracy: {final_accuracy * 100:.2f}%")
print("---------------------------------")
# Example prediction
sample_idx = 0
prediction = final_predictions[sample_idx]
actual = y_test[sample_idx]
print(
    f"Sample prediction: {iris.target_names[prediction]} (Actual: {iris.target_names[actual]})")
