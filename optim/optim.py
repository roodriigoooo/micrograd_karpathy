import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            p.grad_fill(0)

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.mom = momentum
        self.weight_decay = weight_decay
        self._velocity = [np.zeros_like(p.data) for p in self.params]


    def step(self):
        for p, v in zip(self.params, self._velocity):
            if self.weight_decay != 0.0:
                p.grad += self.weight_decay * p.data

            v *= self.mom
            v += p.grad
            p.data -= self.lr * v


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.b1, self.b2 = betas[0], betas[1]
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        b1t = 1 - self.b1 ** self.t
        b2t = 1 - self.b2 ** self.t
        for p, m, v in zip(self.params, self._m, self._v):
            g = p.grad + self.weight_decay * p.data if self.weight_decay else p.grad
            m[:] = self.b1 * m + (1 - self.b1) * g
            v[:] = self.b2 * v + (1 - self.b2) * g ** 2

            m_hat = m / b1t
            v_hat = v / b2t
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    """
    Decoupled weight-decay (Loshchilov & Hutter 2019).
    Only difference: we *subtract* wd term directly from parameters.
    """
    def step(self):
        self.t += 1
        b1t = 1 - self.b1 ** self.t
        b2t = 1 - self.b2 ** self.t
        for p, m, v in zip(self.params, self._m, self._v):
            m[:] = self.b1 * m + (1 - self.b1) * p.grad
            v[:] = self.b2 * v + (1 - self.b2) * (p.grad ** 2)

            m_hat = m / b1t
            v_hat = v / b2t
            p.data -= (
                self.lr * m_hat / (np.sqrt(v_hat) + self.eps) +
                self.lr * self.weight_decay * p.data          # ← decoupled
            )





