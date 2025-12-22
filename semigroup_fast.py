# ===================== megatron/core/transformer/semigroup_attention.py =====================
# Megatron-compatible Semigroup Attention with small-time generator step.
#
# Implements "Approach 1" in a single-step form:
#   - construct a nonnegative, causal Markov kernel P from logits + diffusion prior
#   - interpret L = P - I as a continuous-time generator (rows sum to 0)
#   - approximate P(t) = exp(t L) by the first-order step
#         P(t) ≈ (1 - t) I + t P
#   - apply y = P(t) v = (1 - t) v + t P v
#
# At initialization (with kappa_init = 0, dt_init ≈ 1), this is very close to
# standard softmax attention (Markov kernel on dp), but with a legitimate
# generator interpretation and learnable diffusion prior.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------- RoPE helpers -------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.size(-1)
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    rotary_pos_emb: Optional[Union[Tuple[torch.Tensor, torch.Tensor], object]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q,k: [B,H,T,Dh]
    Supports:
      - (cos, sin) tuple broadcastable to [1,1,T,Dh] / [B,1,T,Dh]
      - objects with apply_rotary_pos_emb(q,k)
      - callable rotary_pos_emb(q,k) -> (q,k)
    """
    if rotary_pos_emb is None:
        return q, k

    if isinstance(rotary_pos_emb, tuple) and len(rotary_pos_emb) == 2:
        cos, sin = rotary_pos_emb

        # broadcast to [1,1,T,Dh] then to [B,H,T,Dh] via broadcasting
        if cos.dim() == 2:  # [T,Dh]
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif cos.dim() == 3:  # [1,T,Dh] or [B,T,Dh]
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        q_out = (q * cos) + (_rotate_half(q) * sin)
        k_out = (k * cos) + (_rotate_half(k) * sin)
        return q_out, k_out

    if hasattr(rotary_pos_emb, "apply_rotary_pos_emb"):
        return rotary_pos_emb.apply_rotary_pos_emb(q, k)

    if callable(rotary_pos_emb):
        out = rotary_pos_emb(q, k)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            return out[0], out[1]

    return q, k


# ------------------------------- Mask helpers -------------------------------

def _mask_to_allowed(attn_mask: Optional[torch.Tensor], B: int, T: int, device) -> Optional[torch.Tensor]:
    """
    Return allowed mask [B,1,T,T] bool where True = allowed.

    Handles:
      - additive masks (0 allowed, -inf/-1e4 masked)
      - bool masks (commonly True=masked in Megatron)
      - 0/1 masks (commonly 1=masked, 0=allowed)
    """
    if attn_mask is None:
        return None

    m = attn_mask
    if m.device != device:
        m = m.to(device)

    if m.dim() == 2:          # [T,T]
        m = m.unsqueeze(0).unsqueeze(0)
    elif m.dim() == 3:        # [B,T,T]
        m = m.unsqueeze(1)
    elif m.dim() != 4:
        raise ValueError(f"attention_mask must have dim 2/3/4, got {tuple(m.shape)}")

    if m.size(-1) != T or m.size(-2) != T:
        raise ValueError(f"attention_mask last dims must be T,T; got {tuple(m.shape)} with T={T}")

    if m.size(0) == 1 and B > 1:
        m = m.expand(B, -1, -1, -1)

    if m.dtype == torch.bool:
        # assume True=masked -> allowed = ~m
        return (~m)

    mf = m.float()
    mn = float(mf.min().detach().cpu())
    mx = float(mf.max().detach().cpu())

    # additive mask: masked ~ very negative
    if mn < -1000.0:
        return (mf > -1000.0)

    # binary mask (most common): 1=masked, 0=allowed
    if mn >= 0.0 and mx <= 1.0:
        return (mf < 0.5)

    # fallback: treat zeros as allowed
    return (mf == 0.0)


def _row_softmax_markov(
    logits: torch.Tensor,            # [B,H,T,T]
    allowed: Optional[torch.Tensor]  # [B,1,T,T] or [B,H,T,T], True = allowed
) -> torch.Tensor:
    """
    Turn logits into a row-stochastic Markov kernel P:

      - mask disallowed entries to -inf
      - subtract rowwise max for stability
      - exp and renormalize rows to sum to 1
    """
    B, H, T, _ = logits.shape
    neg = torch.finfo(logits.dtype).min

    if allowed is None:
        allowed = torch.ones((B, 1, T, T), dtype=torch.bool, device=logits.device)

    # broadcast allowed to [B,H,T,T] if needed
    if allowed.size(1) == 1 and H > 1:
        allowed = allowed.expand(B, H, -1, -1)
    elif allowed.size(1) != H:
        raise ValueError(f"allowed mask shape {allowed.shape} incompatible with logits {logits.shape}")

    logits_masked = logits.masked_fill(~allowed, neg)

    # numerical stability: subtract per-row maxima
    row_max = logits_masked.max(dim=-1, keepdim=True).values
    logits_shift = logits_masked - row_max

    P = torch.exp(logits_shift)
    P = P.masked_fill(~allowed, 0.0)

    Z = P.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    P = P / Z
    return P


# ------------------------------- Optional ESR utilities (unused here) -------------------------------

def _power_iteration_stationary(P: torch.Tensor, iters: int = 15, teleport: float = 0.01) -> torch.Tensor:
    B, H, T, _ = P.shape
    dtype, device = P.dtype, P.device
    if teleport > 0.0:
        U = torch.full((1, 1, T, T), 1.0 / T, device=device, dtype=dtype)
        P_eff = (1.0 - teleport) * P + teleport * U
    else:
        P_eff = P

    pi = torch.full((B, H, T), 1.0 / T, device=device, dtype=dtype)
    for _ in range(iters):
        pi = torch.einsum("bht,bhtt->bht", pi, P_eff)
        pi = pi.clamp_min(0.0)
        pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return pi


def _adjoint_in_l2pi(P: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    pi_i = pi.unsqueeze(-1).clamp_min(1e-12)
    pi_j = pi.unsqueeze(-2).clamp_min(1e-12)
    return (pi_j / pi_i) * P.transpose(-1, -2)


def _esr_reversibilize(
    P: torch.Tensor,
    allowed: Optional[torch.Tensor],
    teleport: float = 0.01,
    pi_iters: int = 15,
    causal_project: bool = True,
) -> torch.Tensor:
    """
    Kept for backwards compatibility with ESR-based variants.
    Not used by the generator-based implementation below.
    """
    pi = _power_iteration_stationary(P, iters=pi_iters, teleport=teleport)
    P_star = _adjoint_in_l2pi(P, pi)
    P_rev = 0.5 * (P + P_star)

    if allowed is not None:
        P_rev = P_rev.masked_fill(~allowed, 0.0)

    if causal_project:
        B, _, T, _ = P_rev.shape
        causal = torch.tril(torch.ones((T, T), device=P_rev.device, dtype=torch.bool)).view(1, 1, T, T)
        if B > 1:
            causal = causal.expand(B, -1, -1, -1)
        P_rev = P_rev.masked_fill(~causal, 0.0)

    Z = P_rev.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return P_rev / Z


# ------------------------------- Config parsing -------------------------------

@dataclass
class SAConfig:
    hidden_size: int
    num_heads: int
    max_seq_len: int
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    causal: bool = True

    # accuracy knobs
    # NOTE: defaults chosen to be as close as possible to standard softmax at init.
    dt_init: float = 0.999        # in (0,1): semigroup time t for P(t)
    dt_learnable: bool = True
    kappa_init: float = 0.0       # start with no diffusion prior (pure dp), let it be learned
    kappa_learnable: bool = True
    xi_init: float = 32.0         # diffusion length scale (only matters once kappa>0)

    # ESR (kept for compatibility, unused here)
    use_esr: bool = False
    esr_teleport: float = 0.01
    esr_pi_iters: int = 15
    esr_causal_project: bool = True

    # kept for compatibility, but not used in this simplified step
    poisson_trunc: int = 2
    use_poisson_mixture: bool = False


def _get_attr(obj: Any, names, default=None):
    for n in names:
        if obj is not None and hasattr(obj, n):
            return getattr(obj, n)
    return default


def _parse_cfg(transformer_config: Optional[Any], kwargs: Dict[str, Any]) -> SAConfig:
    hidden_size = (
        kwargs.get("hidden_size", None)
        or kwargs.get("n_embd", None)
        or _get_attr(transformer_config, ["hidden_size", "hidden_dim", "model_dim", "n_embd"], None)
    )
    if hidden_size is None:
        raise TypeError("Could not determine hidden size (need hidden_size or n_embd or transformer_config.hidden_size).")

    num_heads = (
        kwargs.get("num_heads", None)
        or kwargs.get("n_head", None)
        or kwargs.get("num_attention_heads", None)
        or _get_attr(transformer_config, ["num_attention_heads", "num_heads", "n_head"], None)
    )
    if num_heads is None:
        raise TypeError("Could not determine num_heads (need num_heads/n_head/num_attention_heads).")

    # max seq len: Megatron commonly uses seq_length.
    max_seq_len = (
        kwargs.get("max_seq_len", None)
        or kwargs.get("seq_length", None)
        or kwargs.get("max_position_embeddings", None)
        or _get_attr(transformer_config, ["seq_length", "max_seq_length", "max_position_embeddings"], None)
        or 4096
    )
    max_seq_len = int(max_seq_len)

    attn_dropout = (
        kwargs.get("attn_dropout", None)
        or kwargs.get("attention_dropout", None)
        or _get_attr(transformer_config, ["attention_dropout", "attn_dropout"], 0.0)
    )

    resid_dropout = (
        kwargs.get("resid_dropout", None)
        or kwargs.get("hidden_dropout", None)
        or _get_attr(transformer_config, ["hidden_dropout", "resid_dropout", "dropout"], 0.0)
    )

    dt_init = float(kwargs.get("dt_init", _get_attr(transformer_config, ["sa_dt_init"], SAConfig.dt_init)))
    dt_learnable = bool(kwargs.get("dt_learnable", _get_attr(transformer_config, ["sa_dt_learnable"], True)))

    kappa_init = float(kwargs.get("kappa_init", _get_attr(transformer_config, ["sa_kappa_init"], SAConfig.kappa_init)))
    kappa_learnable = bool(kwargs.get("kappa_learnable", _get_attr(transformer_config, ["sa_kappa_learnable"], True)))

    xi_init = float(kwargs.get("xi_init", _get_attr(transformer_config, ["sa_xi_init"], SAConfig.xi_init)))

    causal = bool(kwargs.get("causal", True))

    use_esr = bool(kwargs.get("use_esr", _get_attr(transformer_config, ["sa_use_esr"], False)))
    esr_teleport = float(kwargs.get("esr_teleport", _get_attr(transformer_config, ["sa_esr_teleport"], 0.01)))
    esr_pi_iters = int(kwargs.get("esr_pi_iters", _get_attr(transformer_config, ["sa_esr_pi_iters"], 15)))
    esr_causal_project = bool(kwargs.get("esr_causal_project", _get_attr(transformer_config, ["sa_esr_causal_project"], True)))

    poisson_trunc = int(kwargs.get("poisson_trunc", _get_attr(transformer_config, ["sa_poisson_trunc"], 2)))
    use_poisson_mixture = bool(kwargs.get("use_poisson_mixture", _get_attr(transformer_config, ["sa_use_poisson_mixture"], False)))

    return SAConfig(
        hidden_size=int(hidden_size),
        num_heads=int(num_heads),
        max_seq_len=max_seq_len,
        attn_dropout=float(attn_dropout),
        resid_dropout=float(resid_dropout),
        causal=causal,
        dt_init=dt_init,
        dt_learnable=dt_learnable,
        kappa_init=kappa_init,
        kappa_learnable=kappa_learnable,
        xi_init=xi_init,
        use_esr=use_esr,
        esr_teleport=esr_teleport,
        esr_pi_iters=esr_pi_iters,
        esr_causal_project=esr_causal_project,
        poisson_trunc=poisson_trunc,
        use_poisson_mixture=use_poisson_mixture,
    )


# ------------------------------- Core attention (B,T,C) -------------------------------

class CausalSemigroupSelfAttentionSelective(nn.Module):
    """
    Core attention in [B,T,C] format.

    Semigroup step:
      - build a Markov kernel P by masked softmax of (dp + diffusion prior)
      - interpret L = P - I as generator
      - apply the first-order semigroup step:
            y = (1 - t) v + t P v
    """

    def __init__(self, transformer_config: Optional[Any] = None, **kwargs):
        super().__init__()
        cfg = _parse_cfg(transformer_config, kwargs)
        self.cfg = cfg

        assert cfg.hidden_size % cfg.num_heads == 0
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads

        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

        self.attn_drop = nn.Dropout(cfg.attn_dropout)
        self.resid_drop = nn.Dropout(cfg.resid_dropout)

        # Preallocated causal mask buffer for Megatron wrappers: [1,1,S,S].
        causal = torch.tril(torch.ones((cfg.max_seq_len, cfg.max_seq_len), dtype=torch.bool))
        self.register_buffer("mask", causal.view(1, 1, cfg.max_seq_len, cfg.max_seq_len), persistent=False)

        # t in (0,1): semigroup time for P(t).
        dt0 = min(max(float(cfg.dt_init), 1e-4), 1.0 - 1e-4)
        self.dt_logit = nn.Parameter(
            torch.tensor(math.log(dt0 / (1.0 - dt0))), requires_grad=cfg.dt_learnable
        )

        # kappa >= 0 (strength of diffusion prior)
        k0 = max(float(cfg.kappa_init), 0.0)
        k_uncon = math.log(math.expm1(k0) + 1e-8) if k0 > 0 else -20.0
        self.kappa_uncon = nn.Parameter(torch.tensor(k_uncon), requires_grad=cfg.kappa_learnable)

        # xi > 0 (diffusion length scale)
        xi0 = max(float(cfg.xi_init), 1e-3)
        xi_uncon = math.log(math.expm1(xi0) + 1e-8)
        self.xi_uncon = nn.Parameter(torch.tensor(xi_uncon), requires_grad=False)

    def forward(
        self,
        x: torch.Tensor,                               # [B,T,C]
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[Union[Tuple[torch.Tensor, torch.Tensor], object]] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        device = x.device
        dtype = x.dtype

        # QKV projections
        qkv = self.qkv(x)  # [B,T,3C]
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,T,Dh]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE
        q, k = apply_rope(q, k, rotary_pos_emb)

        # content scores
        dp = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(self.head_dim))  # [B,H,T,T]

        # diffusion prior in log-space (Toeplitz in position)
        kappa = F.softplus(self.kappa_uncon).to(dtype=dtype)
        xi = F.softplus(self.xi_uncon).to(dtype=dtype)

        idx = torch.arange(T, device=device)
        dist = (idx[:, None] - idx[None, :]).abs().to(dtype)
        xi_t = torch.tensor(max(float(xi.detach().cpu()), 1e-6), device=device, dtype=dtype)
        logK = (-dist / xi_t).view(1, 1, T, T)  # [1,1,T,T]

        # logits = content + diffusion prior (prior inactive at init if kappa≈0)
        logits = dp + kappa * logK  # [B,H,T,T]

        # allowed mask (user mask ∧ causal mask), then force self-attention allowed
        allowed = _mask_to_allowed(attention_mask, B=B, T=T, device=device)

        if self.cfg.causal:
            if T <= self.mask.size(-1):
                causal_allowed = self.mask[..., :T, :T]  # [1,1,T,T], True = allowed
                if B > 1:
                    causal_allowed = causal_allowed.expand(B, -1, -1, -1)
            else:
                causal_allowed = torch.tril(
                    torch.ones((T, T), device=device, dtype=torch.bool)
                ).view(1, 1, T, T)
                if B > 1:
                    causal_allowed = causal_allowed.expand(B, -1, -1, -1)

            allowed = causal_allowed if allowed is None else (allowed & causal_allowed)

        if allowed is None:
            allowed = torch.ones((B, 1, T, T), dtype=torch.bool, device=device)

        # ensure self-attention is always allowed (avoid empty rows / weird renorm)
        eye_bool = torch.eye(T, device=device, dtype=torch.bool).view(1, 1, T, T)
        if B > 1:
            eye_bool = eye_bool.expand(B, -1, -1, -1)
        allowed = allowed | eye_bool

        # build Markov kernel P by masked softmax
        P = _row_softmax_markov(logits, allowed=allowed)  # [B,H,T,T]

        # dropout on P as mild noise; re-normalize rows to keep Markov
        P = self.attn_drop(P)
        row_sums = P.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        P = P / row_sums

        # semigroup time t in (0,1)
        t = torch.sigmoid(self.dt_logit).to(dtype=dtype)

        # first-order semigroup step: y = (1 - t) v + t P v
        ctx = torch.matmul(P, v)          # [B,H,T,Dh]
        y = (1.0 - t) * v + t * ctx       # [B,H,T,Dh]

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.resid_drop(y)
        return y


# ------------------------------- Megatron wrapper (T,B,C) -------------------------------

class SemigroupSelfAttentionMegatron(nn.Module):
    """
    Module Megatron should instantiate from layer specs.
    Ensures semigroup_attn.mask exists and returns (out, bias).
    """
    def __init__(self, transformer_config: Optional[Any] = None, layer_number: int = 1, **kwargs):
        super().__init__()
        self.layer_number = layer_number
        self.semigroup_attn = CausalSemigroupSelfAttentionSelective(transformer_config, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,                 # [T,B,C] (Megatron default)
        attention_mask: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        inference_params: Optional[Any] = None,
        rotary_pos_emb: Optional[Union[Tuple[torch.Tensor, torch.Tensor], object]] = None,
        packed_seq_params: Optional[Any] = None,
        **kwargs,
    ):
        # tolerate [B,T,C] too (some forks do this)
        if hidden_states.dim() != 3:
            raise ValueError(f"hidden_states must be rank-3, got {tuple(hidden_states.shape)}")

        if hidden_states.size(0) != hidden_states.size(1):
            # ambiguous; assume Megatron [T,B,C]
            x = hidden_states.transpose(0, 1).contiguous()  # [B,T,C]
            y = self.semigroup_attn(x, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)
            out = y.transpose(0, 1).contiguous()           # [T,B,C]
        else:
            # still assume [T,B,C] (safe)
            x = hidden_states.transpose(0, 1).contiguous()
            y = self.semigroup_attn(x, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)
            out = y.transpose(0, 1).contiguous()

        return out, None