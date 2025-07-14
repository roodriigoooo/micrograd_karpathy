import numpy as np
from core.engine import Value
from nn.activations import softmax

class Loss:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class MSELoss(Loss):
    def forward(self, pred, target):
        target_val = Value(target.astype(np.float32))
        return ((pred - target_val) ** 2).mean() #broadcasting

class CrossEntropyLoss(Loss):
    def forward(self, logits, targets):
        n = targets.shape[0]
        # for stability, we substract the max
        log_probs = (logits - logits.max(axis=1, keepdims=True))
        log_probs = log_probs - log_probs.exp().sum(axis=1, keepdims=True).log()

        # log prob of gold class
        losses = -log_probs[(np.arange(n), targets)]
        return losses.mean()

class NLLLoss(Loss):
    """
    negative log likelihood loss
    """
    def forward(self, log_probs, targets):
        n = targets.shape[0]
        losses = -log_probs[(np.arange(n), targets)]
        return losses.mean()