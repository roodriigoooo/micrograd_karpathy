from milligrad import *
from graph_utils import draw_dot

class Layer:
    def __init__(self, num_inputs, num_outputs):
        limit = np.sqrt(6 / (num_inputs + num_outputs))
        self.w = Value(np.random.uniform(-limit, limit, (num_inputs, num_outputs)))
        self.b = Value(np.zeros(num_outputs)) #broadcasts during addition

    def __call__(self, x):
        # x is now a Value object containing a batch of inputs
        # e.g., shape (batch_size, num_inputs)
        act = x @ self.w + self.b
        out = act.tanh()
        return out

    def parameters(self):
        return [self.w, self.b]

class MLP:
    def __init__(self, num_inputs, nouts):
        sz = [num_inputs] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        """sets the gradients to zero"""
        for p in self.parameters():
            p.grad.fill(0)

model = MLP(3, [4, 4, 1])
print(f'Network parameters:\n{model.parameters()}')

xs = Value(np.array([
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0,-1.0],
]))

ys = Value(np.array([[1.0], [-1.0], [-1.0], [1.0]]))
lr = 0.03
# Training loop
for k in range(20):
    # Forward pass
    ypred = model(xs)
    # Mean Squared Error Loss
    loss = ((ypred - ys)**2).sum() # Need to implement sum()
    model.zero_grad()
    loss.backward()
    for p in model.parameters():
        p.data -= lr * p.grad

    print(f'iter: {k}, loss: {loss.data}')

print(f'Final predictions {model(xs)}')

draw_dot(loss)

