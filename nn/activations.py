import numpy as np
from core.engine import Value

def tanh(x: Value):
    return x.tanh()

def relu(x: Value):
    return x.relu()

def sigmoid(x: Value):
    s = (1.0 + (-x).exp()) ** -1 #reuse ops
    return s #grad handled automatically

def gelu(x: Value):
    c = (2 / np.pi)**0.5
    #approx form
    return 0.5 * x * (1.0 + (c * (x + 0.044715 * x ** 3)).tanh())

def softmax(x:Value, axis=-1):
    # num stable log-softmax then exp
    shift = x - x.maximum(axis=axis, keepdims=True)
    exps = shift.exp()
    return exps/exps.sum(axis=axis, keepdims=True)