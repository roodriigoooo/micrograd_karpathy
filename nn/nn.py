from core.engine import *


class Module:
    """
    base class for all nn modules.
    all models subclass this class
    like in pytorch
    """
    def zero_grad(self):
        for p in self.parameters():
            p.grad.fill(0)

    def parameters(self):
        params = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Value):
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
            elif isinstance(attr, list) and all(isinstance(m, Module) for m in attr):
                for m in attr:
                    params.extend(m.parameters())
        return list(dict.fromkeys(params))

class Layer(Module):
    def __init__(self, num_inputs, num_outputs, activation='relu'):
        super().__init__()
        limit = np.sqrt(6 / (num_inputs + num_outputs))
        self.w = Value(np.random.uniform(-limit, limit, (num_inputs, num_outputs)))
        self.b = Value(np.zeros(num_outputs)) #broadcasts during addition

        if activation not in [None, 'tanh', 'relu']:
            raise ValueError('Unsupported activation function. Use None, tanh, or relu, or create your own.')
        self.activation = activation

    def __call__(self, x):
        # x is now a Value object containing a batch of inputs
        # e.g., shape (batch_size, num_inputs)
        act = x @ self.w + self.b
        if self.activation == 'tanh':
            return act.tanh()
        elif self.activation == 'relu':
            return act.relu()
        else:
            return act

    def __repr__(self):
        return f"Layer(in={self.w.shape[0]}, out={self.w.shape[1]}, act='{self.activation}')"

    def parameters(self):
        return [self.w, self.b]

class MLP(Module):
    def __init__(self, num_inputs, layer_sizes):
        super().__init__()
        sz = [num_inputs] + layer_sizes
        self.layers = []
        for i in range(len(layer_sizes)):
            activation = 'tanh' if i < len(layer_sizes) -1 else None
            self.layers.append(Layer(sz[i], sz[i+1], activation=activation))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f'MLP of Layers: {self.layers}'



