import math
import collections
from einops import rearrange

import torch
from torch import nn
import torch.nn.functional as F

TTTOperator = collections.namedtuple("TTTOperator", ["start", "end", "update", "apply"])

@torch.no_grad()
def compute_input_coverage_with_qk_and_target_mask(
    q, k,                              # (B*h, L_tot, d)
    ttt_ua_order,                      # TTTOperator list
    input_mask,                        # (B, V_in, 1, H, W)   1=hole, 0=known
    target_mask,                       # (B, V_tar, 1, H, W)  1=hole, 0=known
    patch_size,                        # P
    num_heads,                         # h
):

    Lin_end = ttt_ua_order[0].end
    Ltot_end = ttt_ua_order[-1].end
    Ltar = Ltot_end - Lin_end

    K_in  = k[:, :Lin_end, :]                  # (B*h, Lin,  d)
    Q_tar = q[:, Lin_end:Ltot_end, :]          # (B*h, Ltar, d)

    B, V_in, _, H, W = input_mask.shape
    _, V_tar, _, _, _ = target_mask.shape
    P = patch_size
    Hp, Wp = H // P, W // P
    L_img = Hp * Wp

    R = (1.0 - input_mask.float())             # (B,Vin,1,H,W)
    R_tok = F.avg_pool2d(R.view(B*V_in,1,H,W), P, stride=P)      # (B*Vin,1,Hp,Wp)
    R_tok = R_tok.view(B, V_in, 1, Hp*Wp).transpose(2,3).reshape(B, V_in*L_img, 1)  # (B,Lin,1)
    R_tok = R_tok.repeat_interleave(num_heads, dim=0)            # (B*h,Lin,1)

    Qn = Q_tar / (Q_tar.norm(dim=-1, keepdim=True) + 1e-6)
    Kn = K_in  / (K_in .norm(dim=-1, keepdim=True) + 1e-6)
    A  = torch.softmax(torch.bmm(Qn, Kn.transpose(1,2)), dim=-1) # (B*h,Ltar,Lin)

    conf_token = torch.bmm(A, R_tok).clamp(0,1)                  # (B*h,Ltar,1)

    T = target_mask.float()                                      # (B,Vtar,1,H,W)
    T_tok = F.avg_pool2d(T.view(B*V_tar,1,H,W), P, stride=P)     # (B*Vtar,1,Hp,Wp)
    T_tok = T_tok.view(B, V_tar, 1, Hp*Wp).transpose(2,3)        # (B,Vtar,L_img,1)
    Known_tok = (1.0 - T_tok)                                    # 1=known, 0=hole

    Bh, _, _ = conf_token.shape
    B_from_bh = Bh // num_heads
    conf_view_bh = conf_token.view(Bh, V_tar, L_img, 1)          # (B*h,Vtar,L_img,1)

    conf_view_all_bh   = conf_view_bh.mean(dim=2)                # (B*h,Vtar,1)
    eps = 1e-6
    # known
    kw = Known_tok.repeat_interleave(num_heads, dim=0)           # (B*h,Vtar,L_img,1)
    conf_view_known_bh = (conf_view_bh*kw).sum(2) / (kw.sum(2)+eps)
    # hole
    hw = (1.0 - Known_tok).repeat_interleave(num_heads, dim=0)
    conf_view_hole_bh  = (conf_view_bh*hw).sum(2) / (hw.sum(2)+eps)

    def head_mean(x_bh):
        return x_bh.view(B_from_bh, num_heads, V_tar, 1).mean(1)

    conf_view_all   = head_mean(conf_view_all_bh)
    conf_view_known = head_mean(conf_view_known_bh)
    conf_view_hole  = head_mean(conf_view_hole_bh)

    return conf_token, conf_view_all, conf_view_known, conf_view_hole

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

@torch.compile
# import torch._dynamo
# @torch._dynamo.disable
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
    token_conf: torch.Tensor | None = None,
    conf_min: float = 0.0,
    conf_mode: str = "w0w2",                  # "off" | "w0w2" | "w1" | "all"
    weight_power: float = 0.5,
    num_heads: int = 1,
    collect_stats: bool = False,
    img_tokens_per_view: int | None = None, 
):
    """
    Note:
    Forward:
    (silu(x @ w0) * (x @ w2)) @ w1

    w0, w2: [b, d, dh]
    w1:     [b, dh, d]
    q: [b, l, d]
    k: [b, l, d]
    v: [b, l, d]
    lr0, lr1, lr2: [b, l, 1]
    """
    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)

    output = []
    for start, end, update, apply in ttt_ua_order:
        w0_now, w1_now, w2_now = w0, w1, w2

        if update:
            ki, vi = k[:, start:end, :], v[:, start:end, :]  # bf16
            lr0i = lr0[:, start:end, :]  # [b, l, d/1] fp32
            lr1i = lr1[:, start:end, :]  # [b, l, d/1] fp32
            lr2i = lr2[:, start:end, :]  # [b, l, d/1] fp32

            gate_before_act = ki @ w0_now       # b[b, l, dh] = [b, l, d] @ [b, d, dh]
            hidden_before_mul = ki @ w2_now     # b[b, l, dh] = [b, l, d] @ [b, d, dh]
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

            dhidden = vi @ w1_now.transpose(-1, -2)  # [b, l, dh] = [b, l, d] @ [b, d, dh]
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            qi = q[:, start:end, :]
            align = (qi * ki).sum(dim=-1, keepdim=True)
            beta_from_qk = torch.sigmoid(align) # (B*h, l, 1) in (0,1)
            b = beta_from_qk.pow(0.5) if weight_power is None else beta_from_qk.pow(weight_power)

            if token_conf is not None:
                if len(token_conf.shape) == 2:
                    token_conf = token_conf.unsqueeze(-1)
                conf_u = token_conf[:, start:end, :] 
                conf_smooth = conf_u + (1.0 - conf_u) * 0.1 
                
                lr0i = lr0i * conf_smooth
                lr1i = lr1i * conf_smooth
                lr2i = lr2i * conf_smooth

            H_w  = (hidden * lr1i) * b      ; V_w  = vi * b
            K0_w = (ki * lr0i) * b          ; Dg_w = dgate_before_act * b
            K2_w = (ki * lr2i) * b          ; Dh_w = dhidden_before_mul * b

            w1_grad = zeropower_via_newtonschulz5(H_w.transpose(-1,-2) @ V_w,  muon_update_steps)
            w0_grad = zeropower_via_newtonschulz5(K0_w.transpose(-1,-2) @ Dg_w, muon_update_steps)
            w2_grad = zeropower_via_newtonschulz5(K2_w.transpose(-1,-2) @ Dh_w, muon_update_steps)

            w1_now = w1_now + w1_grad
            w0_now = w0_now + w0_grad
            w2_now = w2_now + w2_grad

            # do weight norm here
            w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
            w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
            w2_now = w2_now / (w2_now.norm(dim=1, keepdim=True) + 1e-5) * w2_norm

            w0, w1, w2 = w0_now, w1_now, w2_now

        if apply:
            # Only calculate the output in the last repeat.
            qi = q[:, start:end, :]
            oi = (F.silu(qi @ w0_now, inplace=True) * (qi @ w2_now)) @ w1_now
            output.append(oi)

    output = torch.cat(output, dim=1)
    return output, w0, w1, w2


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

        self.o_norm = torch.nn.RMSNorm(head_dim, eps=1e-5, elementwise_affine=True)

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
  
        token_conf = None
        if "token_conf" in info:
            conf = info["token_conf"]
            token_conf = conf.repeat_interleave(self.num_heads, dim=0)

        if "w0" in info:
            assert "w1" in info and "w2" in info
            w0 = info["w0"]
            w1 = info["w1"]
            w2 = info["w2"]
        else:
            w0 = self.w0.repeat(x.shape[0], 1, 1)
            w1 = self.w1.repeat(x.shape[0], 1, 1)
            w2 = self.w2.repeat(x.shape[0], 1, 1)

        output, w0, w1, w2 = fast_weight_swish_glu_weight_norm_mini_batch_apply(
            w0, w1, w2, q, k, v, lr0, lr1, lr2, info["ttt_op_order"],
            muon_update_steps=self.muon_update_steps,
            token_conf=token_conf,
            conf_mode="w0w2",
            weight_power=0.5,
            num_heads=self.num_heads,
            img_tokens_per_view=info["num_img_tokens"],
        )
        output = self.o_norm(output)
        output = rearrange(
            output, "(b h) l d -> b l (h d)", h=self.num_heads, b=x.shape[0]
        )

        with torch.no_grad():
            align = (q * k).sum(dim=-1) # [B*h, L]
            beta_map = torch.sigmoid(align)
            beta_map = rearrange(beta_map, "(b h) l -> b h l", h=self.num_heads, b=x.shape[0])
            
            final_lr_map = beta_map.clone()
            if "token_conf" in info:
                conf_u = info["token_conf"].view(x.shape[0], -1) # [B, L]
                conf_smooth = conf_u + (1.0 - conf_u) * 0.1
                final_lr_map = final_lr_map * conf_smooth.unsqueeze(1)

        output = self.c_proj(output)
        ret = {
            "w0": w0, "w1": w1, "w2": w2,
            "beta_map": beta_map,
            "final_lr_map": final_lr_map
        }
        return output, ret

    def extra_repr(self) -> str:
        return (f"w0 shape: {self.w0.shape}, w1 shape: {self.w1.shape}, w2 shape: {self.w2.shape}, "
                f"Muon update steps: {self.muon_update_steps}, "
                f"Base lr: {math.log(1 + math.exp(self.base_lr_inv))}, ")

