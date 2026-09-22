# models/nn_compat.py
import torch
import torch.nn as nn

try:
    from torch.nn import RMSNorm as TorchRMSNorm
    RMSNorm = TorchRMSNorm
except Exception:
    class RMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6, elementwise_affine=True, bias=False,
                    device=None, dtype=None):
            super().__init__()
            self.dim = dim
            self.eps = eps
            self.elementwise_affine = elementwise_affine
            factory = {"device": device, "dtype": dtype}

            if elementwise_affine:
                self.weight = nn.Parameter(torch.ones(dim, **factory))
                self.bias = nn.Parameter(torch.zeros(dim, **factory)) if bias else None
            else:
                self.register_parameter("weight", None)
                self.register_parameter("bias", None)

            self.reset_parameters()  # ← 추가

        def reset_parameters(self):  # ← 추가
            if self.elementwise_affine and self.weight is not None:
                nn.init.ones_(self.weight)
            if getattr(self, "bias", None) is not None:
                nn.init.zeros_(self.bias)

        def forward(self, x):
            x_f = x.float()
            inv_rms = torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            y = (x_f * inv_rms).to(x.dtype)
            if self.elementwise_affine and self.weight is not None:
                y = y * self.weight
                if self.bias is not None:
                    y = y + self.bias
            return y

