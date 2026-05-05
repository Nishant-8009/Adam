"""
optimizer.py — Custom Adam Optimizer (from scratch).

Based on: Kingma & Ba (2015) "Adam: A Method for Stochastic Optimization"
ICLR 2015 - https://arxiv.org/abs/1412.6980
"""

import torch


class AdamOptimizer:
    """
    Custom implementation of the Adam optimizer.

    Algorithm 1 from Kingma & Ba (2015):
      m_t = beta1 * m_{t-1} + (1 - beta1) * g_t          [biased 1st moment]
      v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2        [biased 2nd moment]
      m̂_t = m_t / (1 - beta1^t)                          [bias-corrected 1st moment]
      v̂_t = v_t / (1 - beta2^t)                          [bias-corrected 2nd moment]
      theta_t = theta_{t-1} - alpha * m̂_t / (sqrt(v̂_t) + eps)  [parameter update]

    Default recommended values: alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8
    """

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Args:
            params  : list of torch.Tensor parameters to optimize
            lr      : step size alpha (default: 0.001)
            beta1   : exponential decay rate for 1st moment (default: 0.9)
            beta2   : exponential decay rate for 2nd moment (default: 0.999)
            eps     : numerical stability constant (default: 1e-8)
        """
        self.params = list(params)
        self.lr     = lr
        self.beta1  = beta1
        self.beta2  = beta2
        self.eps    = eps
        self.t      = 0   # global timestep

        # Initialize 1st and 2nd moment vectors to zero for each parameter
        self.m = [torch.zeros_like(p, device=p.device) for p in self.params]
        self.v = [torch.zeros_like(p, device=p.device) for p in self.params]

    def zero_grad(self):
        """Zero out gradients of all tracked parameters."""
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def step(self):
        """
        Perform one Adam update step.
        Implements Eq. (1)–(4) from the paper exactly.
        """
        self.t += 1  # increment global timestep

        # Bias-correction factors (scalar, not per-parameter)
        bc1 = 1.0 - self.beta1 ** self.t   # 1 - beta1^t
        bc2 = 1.0 - self.beta2 ** self.t   # 1 - beta2^t

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad.data  # gradient at current timestep

            # Eq. (1): update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g

            # Eq. (2): update biased second raw moment estimate (element-wise square)
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g

            # Eq. (3): compute bias-corrected moment estimates
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2

            # Eq. (4): update parameters
            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)