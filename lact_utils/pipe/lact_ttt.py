import math
import collections
from einops import rearrange

import torch
from torch import nn
import torch.nn.functional as F

TTTOperator = collections.namedtuple("TTTOperator", ["start", "end", "update", "apply"])

from utils.nn_compat import RMSNorm

@torch.compile
def inv_softplus(x):
    y = x + math.log(-math.expm1(-x))
    return y

@torch.compile
def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """
    Args:
        dy: [b, d, l], gradient of the outer loss wrt the y
        x: [b, d, l], input of the silu activation
    outs:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = dy * sigma * (1 + x * (1 - sigma))
    """
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx

@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py#L49
    Major change: G is [b, d, d] rather than [d, d]
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    Args:
        G: [b, d, d]
        steps: int
    Returns:
        X: [b, d, d]
    """
    assert len(G.shape) == 3
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X

import torch
import torch.nn.functional as F

# NOTE: assumes these exist in your codebase
# - silu_backprop(dy, x)
# - zeropower_via_newtonschulz5(mat, muon_update_steps)

@torch.compile
def fast_weight_swish_glu_weight_norm_mini_batch_apply(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    ttt_ua_order: list,
    muon_update_steps: int = 0,
    lact_original: bool = False,
    chunk_size: int = 2048, 
    eps: float = 1e-5,
):
    """
    Forward:
      (silu(x @ w0) * (x @ w2)) @ w1

    w0, w2: [b, d, dh]
    w1:     [b, dh, d]
    q,k,v:  [b, l, d]
    lr*:    [b, l, 1] (typically fp32)
    """

    # --- keep original norms (detached) for weight norm ---
    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)

    # --- precompute total output length for prealloc (avoid list+cat peak) ---
    total_L = 0
    for start, end, update, apply in ttt_ua_order:
        if apply:
            total_L += (end - start)

    # output dtype: follow q.dtype (usually bf16/fp16)
    B, _, D = q.shape
    out = torch.empty((B, total_L, D), device=q.device, dtype=q.dtype)
    out_pos = 0

    # --- main loop ---
    for start, end, update, apply in ttt_ua_order:
        w0_now, w1_now, w2_now = w0, w1, w2

        # =======================
        # UPDATE (VRAM-optimized)
        # =======================
        if update:
            # Shapes
            dh = w0_now.shape[-1]

            # We will accumulate only the small Gram matrices in fp32:
            # M1: [B, dh, D], M0/M2: [B, D, dh]
            M1 = torch.zeros((B, dh, D), device=q.device, dtype=torch.float32)
            M0 = torch.zeros((B, D, dh), device=q.device, dtype=torch.float32)
            M2 = torch.zeros((B, D, dh), device=q.device, dtype=torch.float32)

            # chunk over tokens to reduce peak activation memory
            cs = chunk_size if chunk_size > 0 else (end - start)
            for s in range(start, end, cs):
                e = s + cs
                if e > end:
                    e = end

                ki = k[:, s:e, :]     # [B, cL, D]
                vi = v[:, s:e, :]     # [B, cL, D]
                # (only needed for beta / lact_original alignment)
                qi = q[:, s:e, :]     # [B, cL, D]

                lr0i = lr0[:, s:e, :] # [B, cL, 1]
                lr1i = lr1[:, s:e, :]
                lr2i = lr2[:, s:e, :]

                # --- compute hidden + grads in chunk ---
                gate_before_act = ki @ w0_now         # [B, cL, dh]
                hidden_before_mul = ki @ w2_now       # [B, cL, dh]

                # avoid inplace in compiled graph for safety
                silu_gate = F.silu(gate_before_act, inplace=False)
                hidden = silu_gate * hidden_before_mul

                dhidden = vi @ w1_now.transpose(-1, -2)  # [B, cL, dh]
                dhidden_before_mul = dhidden * silu_gate
                dgate = dhidden * hidden_before_mul
                dgate_before_act = silu_backprop(dgate, gate_before_act)

                # --- alignment/beta weights (your modified path) ---
                # keep these in q.dtype (bf16) for memory, then multiply
                align = (qi * ki).sum(dim=-1, keepdim=True)   # [B, cL, 1]
                bcoef = torch.sigmoid(align).sqrt()           # [B, cL, 1]

                if lact_original:
                    # original: no extra b weighting (but still chunked accumulation)
                    # IMPORTANT: prevent bf16*fp32 upcast explosion by casting lr to q.dtype first
                    lr1c = lr1i.to(hidden.dtype)
                    lr0c = lr0i.to(ki.dtype)
                    lr2c = lr2i.to(ki.dtype)

                    # accumulate in fp32
                    M1 = M1 + ((hidden * lr1c).transpose(-1, -2) @ vi).to(torch.float32)
                    M0 = M0 + ((ki * lr0c).transpose(-1, -2) @ dgate_before_act).to(torch.float32)
                    M2 = M2 + ((ki * lr2c).transpose(-1, -2) @ dhidden_before_mul).to(torch.float32)
                else:
                    # weighted version: apply lr and bcoef in low precision, accumulate matmul in fp32
                    # cast lr/bcoef to match operand dtype BEFORE multiply (avoid fp32 activation blow-up)
                    lr1c = lr1i.to(hidden.dtype)
                    lr0c = lr0i.to(ki.dtype)
                    lr2c = lr2i.to(ki.dtype)
                    bc_h = bcoef.to(hidden.dtype)
                    bc_k = bcoef.to(ki.dtype)
                    bc_d = bcoef.to(dgate_before_act.dtype)

                    H_w  = (hidden * lr1c) * bc_h
                    V_w  = vi * bc_k
                    K0_w = (ki * lr0c) * bc_k
                    Dg_w = dgate_before_act * bc_d
                    K2_w = (ki * lr2c) * bc_k
                    Dh_w = dhidden_before_mul * bc_h

                    M1 = M1 + (H_w.transpose(-1, -2) @ V_w).to(torch.float32)
                    M0 = M0 + (K0_w.transpose(-1, -2) @ Dg_w).to(torch.float32)
                    M2 = M2 + (K2_w.transpose(-1, -2) @ Dh_w).to(torch.float32)

            # --- one zeropower per segment (huge VRAM win) ---
            w1_grad = zeropower_via_newtonschulz5(M1, muon_update_steps)
            w0_grad = zeropower_via_newtonschulz5(M0, muon_update_steps)
            w2_grad = zeropower_via_newtonschulz5(M2, muon_update_steps)

            # apply update
            w1_now = w1_now + w1_grad.to(w1_now.dtype)
            w0_now = w0_now + w0_grad.to(w0_now.dtype)
            w2_now = w2_now + w2_grad.to(w2_now.dtype)

            # weight norm
            w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + eps) * w0_norm
            w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + eps) * w1_norm
            w2_now = w2_now / (w2_now.norm(dim=1, keepdim=True) + eps) * w2_norm

            # commit
            w0, w1, w2 = w0_now, w1_now, w2_now

        # =======================
        # APPLY (also chunkable)
        # =======================
        if apply:
            segL = end - start
            cs = chunk_size if chunk_size > 0 else segL

            # NOTE: here we must use the latest weights.
            # w0_now/w1_now/w2_now should reflect updated weights if update ran above.
            # If update==False, w*_now are the originals (set at top of loop).
            for s in range(start, end, cs):
                e = s + cs
                if e > end:
                    e = end

                qi = q[:, s:e, :]  # [B, cL, D]

                # (silu(q @ w0) * (q @ w2)) @ w1
                # avoid inplace to reduce compile weirdness
                gate = qi @ w0_now
                hmul = qi @ w2_now
                oi = (F.silu(gate, inplace=False) * hmul) @ w1_now  # [B, cL, D]

                out[:, out_pos:out_pos + (e - s), :] = oi
                out_pos += (e - s)

    # return signature compatible
    if lact_original:
        # In your original code you returned conf_up in lact_original path.
        # That debug path was extremely VRAM/graph-breaking; if you *really* need it,
        # compute it OUTSIDE torch.compile or in a separate non-compiled function.
        conf_up = None
        return out, w0, w1, w2, conf_up
    else:
        return out, w0, w1, w2

class FastWeightGluMLPMultihead(nn.Module):
    """
    On init of fast_weight:

    Let's start with the magnitude of the value.
    value_proj is initialized with uniform distribution with range [-1.0/sqrt(d), 1.0/sqrt(d)]
        x is layernormed. So during init, value is unit norm total (not per head, per head is 1.0/sqrt(num_head))
        After silu, value is around norm of 2.7 per head.  (why? seems wired)

    Then for the fast weight, assume initial lr = 0.
    Then with l2_norm of q,k, input is unit normed.
    if w0 is initialized with kaiming, relu(w0 @ q) is unit normed.
    Then w1 is initialized with kaiming, so w1 @ relu(w0 @ q) is of norm sqrt(2) per head
    Since I compute total norm, it is sqrt(2) * sqrt(num_head), which is around 2.7 for dim=512, num_head=4.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        inter_multi: int = 1,
        bias: bool = False,
        base_lr=0.01,
        muon_update_steps=0,
    ):
        super().__init__()
        self.dim = dim
        assert dim % head_dim == 0
        self.num_heads = dim // head_dim
        self.muon_update_steps = muon_update_steps

        d_in = d_out = head_dim
        d_h = int(head_dim * inter_multi)

        gain = math.sqrt(2)  # for relu activations
        self.w0 = nn.Parameter(
            torch.randn(self.num_heads, d_in, d_h) * gain / math.sqrt(d_in)
        )  # [d_h * num_heads,  d_in]
        self.w1 = nn.Parameter(
            torch.randn(self.num_heads, d_h, d_out) * gain / math.sqrt(d_h)
        )  # [d_in * num_heads,  d_h]
        self.w2 = nn.Parameter(
            torch.randn(self.num_heads, d_in, d_h) * gain / math.sqrt(d_in)
        )  # [d_h * num_heads,  d_in]

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.c_proj = nn.Linear(dim, dim, bias=bias)

        self.lr_dim = self.num_heads
        self.lr_fc = nn.Linear(dim, self.lr_dim * 3)
        self.base_lr_inv = inv_softplus(base_lr)

        # self.o_norm = torch.nn.RMSNorm(head_dim, eps=1e-5, elementwise_affine=True)
        self.o_norm = RMSNorm(head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x: torch.Tensor, info={}, *args):
        """
        x: (b, l, d)
        """
        qkv = F.silu(self.to_qkv(x), inplace=True)  # Silu - Linear
        q, k, v = rearrange(
            qkv, "b l (qkv h d) -> qkv (b h) l d",
            qkv=3, h=self.num_heads
        )
        q = q / (q.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)
        k = k / (k.norm(dim=2, keepdim=True) + 1e-5).to(x.dtype)

        with torch.autocast(device_type="cuda", enabled=False):
            lr = self.lr_fc(x.float())  # [b, l, lr_dim]

        lr = torch.nn.functional.softplus(lr.float() + self.base_lr_inv)
        lr0, lr1, lr2 = rearrange(
            lr, "b l (lrs h d) -> lrs (b h) l d",
            lrs=3, h=self.num_heads
        )
  
        if "w0" in info:
            assert "w1" in info and "w2" in info
            w0 = info["w0"]
            w1 = info["w1"]
            w2 = info["w2"]
        else:
            w0 = self.w0.repeat(x.shape[0], 1, 1)
            w1 = self.w1.repeat(x.shape[0], 1, 1)
            w2 = self.w2.repeat(x.shape[0], 1, 1)

        if info['lact_original']:
            output, w0, w1, w2, conf_up = fast_weight_swish_glu_weight_norm_mini_batch_apply(
                w0, w1, w2, q, k, v, lr0, lr1, lr2, info["ttt_op_order"],
                muon_update_steps=self.muon_update_steps, lact_original=info['lact_original']
            )   
            output = self.o_norm(output)
            output = rearrange(
                output, "(b h) l d -> b l (h d)", h=self.num_heads, b=x.shape[0]
            )

            output = self.c_proj(output)
            return output, {"w0": w0, "w1": w1, "w2": w2, "conf": conf_up}                     
        else:
            output, w0, w1, w2 = fast_weight_swish_glu_weight_norm_mini_batch_apply(
                w0, w1, w2, q, k, v, lr0, lr1, lr2, info["ttt_op_order"],
                muon_update_steps=self.muon_update_steps, lact_original=info['lact_original']
            )
            output = self.o_norm(output)
            output = rearrange(
                output, "(b h) l d -> b l (h d)", h=self.num_heads, b=x.shape[0]
            )

            output = self.c_proj(output)
            return output, {"w0": w0, "w1": w1, "w2": w2}

    def extra_repr(self) -> str:
        return (f"w0 shape: {self.w0.shape}, w1 shape: {self.w1.shape}, w2 shape: {self.w2.shape}, "
                f"Muon update steps: {self.muon_update_steps}, "
                f"Base lr: {math.log(1 + math.exp(self.base_lr_inv))}, ")

