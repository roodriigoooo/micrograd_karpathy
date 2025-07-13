from graphviz import Digraph
import numpy as np
import math
from engine import Value

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    """
    modified to handle both scalars and tensors
    """
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
    nodes, edges = trace(root)
    for node in nodes:
        uid = str(id(node))
        if node.data.size == 1:
            label_text = "{ %s | data %.4f | grad %.4f }" % (node.label, node.data, node.grad)
        else:
            label_text = "{ %s | data %s | grad mean: %.4f}" % (node.label or '', str(node.data.shape), np.mean(node.grad))

        dot.node(name=uid, label=label_text, shape='record')

        if node._op:
            dot.node(name=uid + node._op, label=node._op)
            dot.edge(uid + node._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

def numerical_gradient(f, x, eps = 1e-2):
    orig_dtype = x.data.dtype
    grad = np.zeros_like(x.data, dtype=np.float32)

    it = np.nditer(x.data, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        orig_val = float(x.data[idx])

        x.data[idx] = np.float32(orig_val + eps)
        f_plus = float(f(x).data)

        x.data[idx] = np.float32(orig_val - eps)
        f_minus = float(f(x).data)

        x.data[idx] = np.float32(orig_val)

        if math.isfinite(f_plus) and math.isfinite(f_minus):
            grad[idx] = (f_plus - f_minus) / (2 * eps)
        elif math.isfinite(f_plus):
            f0 = float(f(x).data)
            grad[idx] = (f_plus - f0) / (eps)
        elif math.isfinite(f_minus):
            f0 = float(f(x).data)
            grad[idx] = (f0 - f_minus) / (eps)
        else:
            grad[idx] = np.nan

        it.iternext()
    return grad.astype(orig_dtype, copy=False)
          # array([0.738..., nan], dtype=float32)
