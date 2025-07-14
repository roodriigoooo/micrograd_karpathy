from functools import wraps
import numpy as np

def _ensure_value(func):
    """
    a decorator to automatically wrap the 'other' argument in a Val object
    """
    @wraps(func)
    def wrapper(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return func(self, other)
    return wrapper

class Value:
    """
    A scalar or tensor value with automatic differentiation.
    Support broadcasting and gradient un-broadcasting.
    """
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = np.array(data, dtype=np.float32) if not isinstance(data, Value) else data.data
        self.grad = np.zeros_like(self.data, dtype=np.float32)
        self._backward = lambda: None #placeholder
        self._prev = set(_children)
        self._op = _op # the operation that produced this node
        self.label = label
        self.shape = self.data.shape
        self._topo_cache = None

    # util for broadcasting gradients
    @staticmethod
    def _unbroadcast(grad, shape):
        """
        undo numpy broadcasting by summing grad along axes where
        the original shape introduced size-1 dimensions or extra
        leading dimensions
        """
        # sum out broadcasted dimensions
        #remove extra dims
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)

        #sum along dims where original shape == 1
        for ax, s in enumerate(shape):
            if s == 1:
                grad = grad.sum(axis=ax, keepdims=True)
        return grad

    def __repr__(self):
        return f'Value(data={self.data})'

    # -- Binary operations -----------------------
    @_ensure_value
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += Value._unbroadcast(out.grad, self.shape) # chain rule: dL/dx = dL/dout * dout/dx =  out.grad * 1.0
            other.grad += Value._unbroadcast(out.grad, other.shape) # chain rule: dL/dy = dL/dout * dout/dy = out.grad * 1.0
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    @_ensure_value
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    @_ensure_value
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += Value._unbroadcast(other.data * out.grad, self.shape)# Chain rule: dL/dx = dL/dout * dout/dx = out.grad * y
            other.grad += Value._unbroadcast(self.data * out.grad, other.shape) #dL/dy = dL/dout * dout/dy = out.grad * x
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self*other

    @_ensure_value
    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __pow__(self, other):
        if not isinstance(other, Value):
            out = Value(self.data ** other, (self, ), f'**{other}')
            def _backward():
                self.grad += Value._unbroadcast(other * self.data ** (other-1) * out.grad, self.shape)
            out._backward = _backward
            return out

        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data ** other.data, (self, other), f'**{other.data}')

        def _backward():
            # derivative wrt x: y * x^(y-1)
            self.grad += Value._unbroadcast(other.data * (self.data**(other.data-1)) * out.grad, self.shape)
            # derivative wrt y: ln(x) * x^y
            other.grad += Value._unbroadcast(np.log(self.data) * out.data * out.grad, other.shape)
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    @_ensure_value
    # matmul for tensor operations
    def __matmul__(self, other):
        out = Value(self.data @ other.data, (self, other), '@')

        def _backward():
            #promote 1d operands to 2d row/col views so shapes always match
            A = self.data if self.data.ndim > 1 else self.data.reshape(1, -1) #(1, k)
            B = other.data if other.data.ndim > 1 else other.data.reshape(1, -1) #(k, 1)
            dC = out.grad if out.grad.ndim > 1 else out.grad.reshape(1, -1) #(1,m) or (n, m)

            # dL/dA = dL/dC @ B.T
            self_contrib = dC @ B.T
            # dL/dB = A.T @ dL/dC
            other_contrib = A.T @ dC

            self.grad += Value._unbroadcast(self_contrib.reshape(self.shape), self.shape)
            other.grad += Value._unbroadcast(other_contrib.reshape(other.shape), other.shape)

        out._backward = _backward
        return out

    def __rmatmul__(self, other):
        return Value(other) @ self

    # ---- Unary operations -----------------------------
    def exp(self):
        out = Value(np.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(np.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out

    def sin(self):
        out = Value(np.sin(self.data), (self,), 'sin')
        def _backward():
            self.grad += np.cos(self.data) * out.grad
        out._backward = _backward
        return out

    def cos(self):
        out = Value(np.cos(self.data), (self,), 'cos')
        def _backward():
            self.grad += -np.sin(self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        # since max(x, 0) is already diffable via .maximum
        return self.maximum(0) #0 is autowrapped by @_ensure_value dec

    def softplus(self):
        # softplus(x) = log(1+exp(x))
        # all three ops (exp, +, log) are native Value ops
        return (self.exp() + 1).log() # again, auto-wrapped

    def max(self, axis=None, keepdims=False):
        """
        if there a tie for the max, the incoming grad is divided equally among them
        note: this is different from how torch behaves, which propagates gradients only
        to the first element encountered with that maximum value
        """
        out_data = self.data.max(axis=axis, keepdims=keepdims)
        out     = Value(out_data, (self,), 'max_reduce')
        def _backward():
            mask = (self.data == out.data)
            count = mask.sum(axis=axis, keepdims=True)
            grad_broadcast = np.broadcast_to(out.grad, self.shape)
            self.grad += mask * (grad_broadcast / count)

        out._backward = _backward
        return out

    # Elementwise max/min
    @_ensure_value
    def maximum(self, other):
        out_data = np.maximum(self.data, other.data)
        out = Value(out_data, (self, other), 'max')
        def _backward():
            mask_self = (self.data >= other.data)
            mask_other = ~mask_self
            self.grad += mask_self * out.grad
            other.grad += mask_other * out.grad
        out._backward = _backward
        return out

    @_ensure_value
    def minimum(self, other):
        out_data = np.minimum(self.data, other.data)
        out = Value(out_data, (self, other), 'min')
        def _backward():
            mask_self = (self.data <= other.data)
            mask_other = ~mask_self
            self.grad += mask_self * out.grad
            other.grad += mask_other * out.grad
        out._backward = _backward
        return out

    # --- Tensor manipulation ------------
    def mean(self, axis=None, keepdims=False):
        out_data = self.data.mean(axis=axis, keepdims=keepdims)
        out = Value(out_data, (self,), 'mean')

        if axis is None:
            count = self.data.size
            red_axes = tuple(range(self.data.ndim))
        else:
            red_axes = axis if isinstance(axis, tuple) else (axis,)
            #handle negative axis
            red_axes = tuple(ax if ax >= 0 else ax + self.data.ndim for ax in red_axes)
            count = np.prod([self.data.shape[ax] for ax in red_axes], dtype=np.int32)

        def _backward():
            grad = out.grad / count

            if not keepdims:
                shape = list(out.grad.shape)
                for ax in sorted(red_axes):
                    shape.insert(ax,1)
                grad = grad.reshape(shape)

            self.grad += np.broadcast_to(grad, self.shape)

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Value(out_data, (self,), 'sum')

        def _backward():
            # the gradient is just broadcasted to the original shape
            self.grad += np.broadcast_to(out.grad, self.shape)
        out._backward = _backward
        return out

    def reshape(self, new_shape):
        out = Value(self.data.reshape(new_shape), (self,), 'reshape')

        def _backward():
            # gradient is just reshaped back to the original shape
            self.grad += out.grad.reshape(self.shape)
        out._backward = _backward
        return out

    def transpose(self, axes=None):
        out_data = self.data.T if axes is None else np.transpose(self.data, axes)
        out = Value(out_data, (self,), 'transpose')

        def _backward():
            if axes is None:
                self.grad += out.grad.T
            else:
                inv_axes = np.argsort(axes)
                self.grad += np.transpose(out.grad, inv_axes)
        out._backward = _backward
        return out

    def __getitem__(self, key):
        out = Value(self.data[key], (self, ), 'slice')
        def _backward():
            # create a zero grad of the same shape as the original tensor
            # and add the incoming gradient to the sliced region
            new_grad = np.zeros_like(self.data)
            new_grad[key] = out.grad
            self.grad += new_grad
        out._backward = _backward
        return out

    # --- Graph utils --------------------------

    def _build_topo(self, visited=None, topo=None):
        if visited is None: visited = set()
        if topo is None: topo = []
        if self not in visited:
            visited.add(self)
            for child in self._prev:
                child._build_topo(visited, topo)
            topo.append(self)
        return topo

    def backward(self, retain_graph=False):
        """
        backprop gradients from this node to all dependencies.
        clear_graph clears _prev and _backward of all nodes, just
        to free up memory.
        """
        # build/reuse topological order
        if self._topo_cache is None:
            self._topo_cache = self._build_topo()

        # the zeroing of gradients is now handled separately
        self.grad = np.ones_like(self.data, dtype=np.float32)
        # backprop
        for node in reversed(self._topo_cache):
            node._backward()

        # optionally clear graph references
        if not retain_graph:
            for node in self._topo_cache:
                node._prev = set()
                node._backward = lambda : None
            self._topo_cache = None

# # -- Gradient-checking utility, following the definition of a derivative ------
# def gradient_check(func, inputs, eps=1e-6, tol=1e-3):
#     """
#     to numerically verify gradients via a centered difference
#     :param func: function mapping named Value inputs into a single Value output.
#     :param inputs: dict of name -> Value instances.
#     :param eps: perturbation magnitude (h)
#     :param tol: tolerance for comparison
#     :return: (analytical_grads, numeric_grads_), with pass/fail per input
#     """
#     output = func(**inputs)
#     output.backward(clear_graph=True)
#     analytical = {name: inp.grad.copy() for name, inp in inputs.items()}
#
#     # compute numeric gradients
#     numeric = {}
#     for name, inp in inputs.items():
#         orig = inp.data.copy()
#         inp.data = orig + eps
#         f_plus = func(**inputs).data
#         inp.data = orig-eps
#         f_minus = func(**inputs).data
#         inp.data = orig
#         numeric[name] = (f_plus - f_minus) / (2*eps)
#
#     # compare
#     for name in inputs:
#         a, n = analytical[name], numeric[name]
#         if not np.allclose(a, n, atol=tol):
#             print(f'[FAIL] {name}: analytic={a}, numeric={n}')
#         else:
#             print(f'[PASS] {name}: analytic={a}, numeric={n}')
#     return analytical, numeric
