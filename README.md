# milligrad

This is a little project to build a tiny automatic differentiation (autograd) engine from scratch. it is an extension of [micrograd](https://github.com/karpathy/micrograd).

my goal is just to understand at a deep level the mechanics of backpropagation.

some of the modifications I have implemented include transforming micrograd from a scalar-only toolkit into a tensor-aware one. it extends the engine to arbitrary shapes, reduces duplication and has some slight, basic modifications to optimize runtime and memory usage.

## example usage

```python
from core.engine import Value
from nn.nn import MLP
from optim.optim import Adam

# Create a simple neural network
model = MLP(num_inputs=3, layer_sizes=[4, 4, 1])
optimizer = Adam(model.parameters(), lr=0.001)

# Training data
x = Value([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = Value([[1.0], [0.0]])

# Forward pass
pred = model(x)
loss = ((pred - y) ** 2).mean()

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Loss: {loss.data}")
```


### credits and huge thanks

needless to say that this project (among others) would not exist without the work of **Andrej Karpathy**. it is a direct result of following his great ["The spelled-out intro to neural networks and backpropagation: bulding micrograd"](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1). all the core ideas and are his.

if you haven't, please go watch all of his videos. they are the best, clear, intuitive and hands-on explanations  of just about anything happening under the hood of deep learning.

**thank you andrej** :goat: :goat:
