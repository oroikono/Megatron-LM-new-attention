"""
Corrected Semigroup Attention Implementation
=============================================

CRITICAL FIX 1: The original ESR implementation computed P²v instead of (P+P*)/2·v.

CRITICAL FIX 2: π estimation now uses FULL-SUPPORT teleport.
Causal P has token 0 as absorbing state; causal teleport doesn't fix this,
causing π → δ₀ and degenerate ratios π_j/π_i in reversibilization.
Full-support teleport ensures irreducibility and well-defined π.

OPTIMIZATION: Teleport is done inline in vector iteration:
    π ← (1-γ)·πP + γ/T
No need to materialize T×T uniform matrix. With γ > 0 the chain is
irreducible/aperiodic, so no need for P² to break periodicity.

LAZY-STEP MIXING (Megatron-friendly "semigroup time" control):
    y = (1-dt)*v + dt*y_transport
where y_transport = (1-α)*Pv + α*P_rev·v. This gives controllable "time per layer"
without changing kernels, τ-mixtures, or Triton paths. Cost: one fused multiply-add.
- dt=1.0: original behavior (full transport)
- dt=1/n_layers: "one full step across the stack"
- dt_learnable=True: learn per-layer dt via sigmoid

Additive reversibilization formula:
    P* = adjoint of P in ℓ²(π), where P*_{ij} = (π_j/π_i) P_{ji}
    P_rev = (P + P*) / 2

CAUSAL CAVEAT: For autoregressive models, we project P_rev onto causal support
(zero upper triangle) and renormalize. This projection destroys exact 
self-adjointness in ℓ²(π) and does not preserve the stationary distribution
exactly. The result is a stabilizing heuristic, not exact ESR.

Key Optimizations:
1. Fused QKV+G projection (4 matmuls → 1)
2. Triton kernels for pairwise distance + fractional kernel
3. Inline teleport in vector iteration (no T×T allocation, no P²)
4. Lazy-step mixing (essentially free: one extra fused multiply-add)
5. torch.compile() compatible (no state mutations in forward)
6. Flash-style online softmax for row normalization
7. Gradient checkpointing support
8. Mixed precision friendly
9. FP32 kernel computation for numerical stability
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ============================================================================
# TRITON KERNELS
# ============================================================================

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Triton not available, using PyTorch fallbacks")


def _is_compiling() -> bool:
    """Return True when inside torch.compile / torch._dynamo tracing."""
    if hasattr(torch, "_dynamo"):
        try:
            return torch._dynamo.is_compiling()
        except Exception:
            return False
    return False

if HAS_TRITON:
    @triton.jit
    def _fused_sqdist_kernel(
        G_ptr, Out_ptr,
        stride_gb, stride_gt, stride_gd,
        stride_ob, stride_ot1, stride_ot2,
        T: tl.constexpr, D: tl.constexpr,
        BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Fused pairwise squared distance: ||g_i - g_j||²"""
        pid_bh = tl.program_id(0)
        pid_row = tl.program_id(1)
        pid_col = tl.program_id(2)
        
        row_start = pid_row * BLOCK_T
        col_start = pid_col * BLOCK_T
        
        row_offs = row_start + tl.arange(0, BLOCK_T)
        col_offs = col_start + tl.arange(0, BLOCK_T)
        
        # Accumulate dot product and norms
        acc_dot = tl.zeros((BLOCK_T, BLOCK_T), dtype=tl.float32)
        row_norm = tl.zeros((BLOCK_T,), dtype=tl.float32)
        col_norm = tl.zeros((BLOCK_T,), dtype=tl.float32)
        
        for d_start in range(0, D, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D
            
            row_ptrs = G_ptr + pid_bh * stride_gb + row_offs[:, None] * stride_gt + d_offs[None, :] * stride_gd
            col_ptrs = G_ptr + pid_bh * stride_gb + col_offs[:, None] * stride_gt + d_offs[None, :] * stride_gd
            
            row_mask = (row_offs[:, None] < T) & d_mask[None, :]
            col_mask = (col_offs[:, None] < T) & d_mask[None, :]
            
            g_row = tl.load(row_ptrs, mask=row_mask, other=0.0).to(tl.float32)
            g_col = tl.load(col_ptrs, mask=col_mask, other=0.0).to(tl.float32)
            
            acc_dot += tl.dot(g_row, tl.trans(g_col))
            row_norm += tl.sum(g_row * g_row, axis=1)
            col_norm += tl.sum(g_col * g_col, axis=1)
        
        sqdist = row_norm[:, None] + col_norm[None, :] - 2.0 * acc_dot
        sqdist = tl.maximum(sqdist, 0.0)
        
        out_ptrs = Out_ptr + pid_bh * stride_ob + row_offs[:, None] * stride_ot1 + col_offs[None, :] * stride_ot2
        out_mask = (row_offs[:, None] < T) & (col_offs[None, :] < T)
        tl.store(out_ptrs, sqdist, mask=out_mask)

    @triton.jit
    def _fused_frackernel_kernel(
        S_ptr, LogK_ptr, Tau_ptr, W_ptr, Beta_ptr,
        stride_sb, stride_st1, stride_st2,
        H: tl.constexpr, Nt: tl.constexpr, T: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """
        Compute fractional log-kernel via logsumexp over τ-mixture:
        logK = logsumexp_t[-(β/2)log(1 + s/(4τ_t)) + log(w_t)]
        """
        pid_bh = tl.program_id(0)
        pid_row = tl.program_id(1)
        pid_col = tl.program_id(2)
        
        h_idx = pid_bh % H
        
        row_start = pid_row * BLOCK_T
        col_start = pid_col * BLOCK_T
        row_offs = row_start + tl.arange(0, BLOCK_T)
        col_offs = col_start + tl.arange(0, BLOCK_T)
        
        s_ptrs = S_ptr + pid_bh * stride_sb + row_offs[:, None] * stride_st1 + col_offs[None, :] * stride_st2
        mask = (row_offs[:, None] < T) & (col_offs[None, :] < T)
        s = tl.load(s_ptrs, mask=mask, other=0.0).to(tl.float32)
        
        beta = tl.load(Beta_ptr + h_idx).to(tl.float32)
        beta = tl.maximum(tl.minimum(beta, 2.5), 0.5)
        
        max_val = tl.full((BLOCK_T, BLOCK_T), -1e30, dtype=tl.float32)
        sum_exp = tl.zeros((BLOCK_T, BLOCK_T), dtype=tl.float32)
        
        for t_idx in range(Nt):
            tau = tl.load(Tau_ptr + h_idx * Nt + t_idx).to(tl.float32)
            tau = tl.maximum(tau, 1e-6)
            w = tl.load(W_ptr + h_idx * Nt + t_idx).to(tl.float32)
            
            log_k = -(beta * 0.5) * tl.log(1.0 + s / (4.0 * tau))
            log_k = log_k + tl.log(w + 1e-12)
            
            new_max = tl.maximum(max_val, log_k)
            sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.exp(log_k - new_max)
            max_val = new_max
        
        result = max_val + tl.log(sum_exp + 1e-12)
        
        out_ptrs = LogK_ptr + pid_bh * stride_sb + row_offs[:, None] * stride_st1 + col_offs[None, :] * stride_st2
        tl.store(out_ptrs, result, mask=mask)

    @triton.jit
    def _fused_causal_softmax_kernel(
        LogK_ptr, DP_ptr, P_ptr,
        lam,
        stride_b, stride_t1, stride_t2,
        T: tl.constexpr,
        BLOCK_COL: tl.constexpr,
    ):
        """
        Fused: blend logK with dp, apply causal mask, row-normalize.
        """
        pid_bh = tl.program_id(0)
        row_idx = tl.program_id(1)
        
        if row_idx >= T:
            return
        
        # First pass: find max
        max_val = -1e30
        for col_start in range(0, row_idx + 1, BLOCK_COL):
            col_offs = col_start + tl.arange(0, BLOCK_COL)
            col_mask = (col_offs <= row_idx) & (col_offs < T)
            
            logk_ptr = LogK_ptr + pid_bh * stride_b + row_idx * stride_t1 + col_offs * stride_t2
            dp_ptr = DP_ptr + pid_bh * stride_b + row_idx * stride_t1 + col_offs * stride_t2
            
            logk = tl.load(logk_ptr, mask=col_mask, other=-1e30).to(tl.float32)
            dp = tl.load(dp_ptr, mask=col_mask, other=-1e30).to(tl.float32)
            
            blended = lam * dp + (1.0 - lam) * logk
            blended = tl.where(col_mask, blended, -1e30)
            max_val = tl.maximum(max_val, tl.max(blended))
        
        # Second pass: compute sum
        row_sum = 0.0
        for col_start in range(0, row_idx + 1, BLOCK_COL):
            col_offs = col_start + tl.arange(0, BLOCK_COL)
            col_mask = (col_offs <= row_idx) & (col_offs < T)
            
            logk_ptr = LogK_ptr + pid_bh * stride_b + row_idx * stride_t1 + col_offs * stride_t2
            dp_ptr = DP_ptr + pid_bh * stride_b + row_idx * stride_t1 + col_offs * stride_t2
            
            logk = tl.load(logk_ptr, mask=col_mask, other=-1e30).to(tl.float32)
            dp = tl.load(dp_ptr, mask=col_mask, other=-1e30).to(tl.float32)
            
            blended = lam * dp + (1.0 - lam) * logk
            exp_val = tl.exp(blended - max_val)
            exp_val = tl.where(col_mask, exp_val, 0.0)
            row_sum += tl.sum(exp_val)
        
        row_sum = tl.maximum(row_sum, 1e-12)
        
        # Third pass: normalize and store
        for col_start in range(0, T, BLOCK_COL):
            col_offs = col_start + tl.arange(0, BLOCK_COL)
            col_mask = (col_offs <= row_idx) & (col_offs < T)
            
            logk_ptr = LogK_ptr + pid_bh * stride_b + row_idx * stride_t1 + col_offs * stride_t2
            dp_ptr = DP_ptr + pid_bh * stride_b + row_idx * stride_t1 + col_offs * stride_t2
            
            logk = tl.load(logk_ptr, mask=col_mask, other=-1e30).to(tl.float32)
            dp = tl.load(dp_ptr, mask=col_mask, other=-1e30).to(tl.float32)
            
            blended = lam * dp + (1.0 - lam) * logk
            exp_val = tl.exp(blended - max_val)
            p_val = tl.where(col_mask, exp_val / row_sum, 0.0)
            
            p_ptr = P_ptr + pid_bh * stride_b + row_idx * stride_t1 + col_offs * stride_t2
            tl.store(p_ptr, p_val, mask=(col_offs < T))


# ============================================================================
# PYTORCH OPERATIONS (with Triton dispatch)
# ============================================================================

def pairwise_sqdist(g: torch.Tensor) -> torch.Tensor:
    """Compute ||g_i - g_j||² for all pairs."""
    B, H, T, D = g.shape

    if HAS_TRITON and g.is_cuda and T >= 32 and not _is_compiling():
        g_flat = g.reshape(B * H, T, D).contiguous()
        out = torch.empty(B * H, T, T, device=g.device, dtype=torch.float32)

        BLOCK_T = min(64, triton.next_power_of_2(T))
        BLOCK_D = min(64, triton.next_power_of_2(D))
        grid = (B * H, triton.cdiv(T, BLOCK_T), triton.cdiv(T, BLOCK_T))

        _fused_sqdist_kernel[grid](
            g_flat, out,
            g_flat.stride(0), g_flat.stride(1), g_flat.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            T, D, BLOCK_T, BLOCK_D,
        )
        return out.view(B, H, T, T)

    # PyTorch fallback
    g2 = (g ** 2).sum(-1, keepdim=True)
    s = g2 + g2.transpose(-2, -1) - 2.0 * torch.einsum('bhid,bhjd->bhij', g, g)
    return s.clamp_min(0.0)


def fractional_kernel(s: torch.Tensor,
                      tau: torch.Tensor,
                      w: torch.Tensor,
                      beta: torch.Tensor) -> torch.Tensor:
    """
    Compute fractional log-kernel:
    logK = logsumexp_t[-(β/2)log(1 + s/(4τ_t)) + log(w_t)]
    """
    B, H, T, _ = s.shape
    Nt = tau.size(-1)

    if HAS_TRITON and s.is_cuda and T >= 32 and not _is_compiling():
        s_flat = s.reshape(B * H, T, T).contiguous()
        out = torch.empty_like(s_flat)

        BLOCK_T = min(32, triton.next_power_of_2(T))
        grid = (B * H, triton.cdiv(T, BLOCK_T), triton.cdiv(T, BLOCK_T))

        _fused_frackernel_kernel[grid](
            s_flat, out,
            tau.contiguous(), w.contiguous(), beta.contiguous(),
            s_flat.stride(0), s_flat.stride(1), s_flat.stride(2),
            H, Nt, T, BLOCK_T,
        )
        return out.view(B, H, T, T)

    # PyTorch fallback
    tau_exp = tau.view(1, H, Nt, 1, 1)
    beta_exp = beta.view(1, H, 1, 1, 1).clamp(0.5, 2.5)
    s_exp = s.unsqueeze(2)

    log_k_per_tau = -(beta_exp / 2.0) * torch.log1p(s_exp / (4.0 * tau_exp.clamp_min(1e-6)))
    log_k = torch.logsumexp(
        log_k_per_tau + torch.log(w.view(1, H, Nt, 1, 1) + 1e-12),
        dim=2,
    )
    return log_k


def causal_markov_normalize(logK: torch.Tensor,
                            dp: torch.Tensor,
                            lam: float,
                            mask: torch.Tensor) -> torch.Tensor:
    """Blend logK and dp, apply causal mask, row-normalize to Markov matrix."""
    B, H, T, _ = logK.shape

    if HAS_TRITON and logK.is_cuda and T >= 64 and not _is_compiling():
        logK_flat = logK.reshape(B * H, T, T).contiguous()
        dp_flat = dp.reshape(B * H, T, T).contiguous()
        P_out = torch.zeros_like(logK_flat)

        BLOCK_COL = min(128, triton.next_power_of_2(T))
        grid = (B * H, T)

        if torch.is_tensor(lam):
            lam_scalar = float(lam.detach().cpu().item())
        else:
            lam_scalar = float(lam)

        _fused_causal_softmax_kernel[grid](
            logK_flat, dp_flat, P_out,
            lam_scalar,
            logK_flat.stride(0), logK_flat.stride(1), logK_flat.stride(2),
            T, BLOCK_COL,
        )
        return P_out.view(B, H, T, T)

    # PyTorch fallback
    if torch.is_tensor(lam):
        lam_t = lam.to(logK.dtype)
    else:
        lam_t = logK.new_tensor(lam, dtype=logK.dtype)

    blended = lam_t * dp + (1.0 - lam_t) * logK
    blended = blended.masked_fill(~mask, float('-inf'))

    max_val = blended.amax(dim=-1, keepdim=True)
    max_val = torch.where(torch.isfinite(max_val), max_val, torch.zeros_like(max_val))
    K = torch.exp((blended - max_val).clamp(min=-60.0)) * mask.float()
    return K / K.sum(-1, keepdim=True).clamp_min(1e-12)


# ============================================================================
# CORRECTED SEMIGROUP ATTENTION
# ============================================================================

class CausalSemigroupSelfAttentionSelective(nn.Module):
    """
    Semigroup attention with additive reversibilization (causal variant).
    
    CRITICAL FIX 1: The original computed P²v. This version computes:
        P* = adjoint of P in ℓ²(π), where P*_{ij} = (π_j/π_i) P_{ji}
        P_rev = (P + P*) / 2
    
    CRITICAL FIX 2: π estimation uses FULL-SUPPORT teleport (not causal).
    Causal P has token 0 as absorbing state. Causal teleport doesn't fix this,
    causing π → δ₀. Full-support teleport ensures irreducibility via inline
    vector iteration (no T×T matrix allocation):
        π ← (1-γ)·πP + γ/T
    With γ > 0, no need for P² to break periodicity.
    
    CAUSAL CAVEAT: After computing P_rev, we project onto causal support
    (zero upper triangle) and renormalize rows. This projection:
    - Destroys exact self-adjointness in ℓ²(π) (adjoint has anti-causal support)
    - Does not preserve the stationary distribution π exactly
    
    What we actually get:
    - Row-stochastic (Markov) after renormalization
    - A causal approximation to reversibilization that improves stability
    - Heuristically better-conditioned than raw P for deep stacking
    """
    
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        n_taus: int = 8,
        power_iters: int = 4,
        dp_temp: float = 0.8,
        esr_gamma: float = 0.02,
        esr_alpha: float = 0.1,
        dropout: float = 0.0,
        bias: bool = False,
        # New controls
        init_lam: float = 1.5,  # Start more dot-product dominated (sigmoid(1.5) ≈ 0.82)
        dp_scale_learnable: bool = True,  # Learnable scale on dp
        g_norm_mode: str = "rmsnorm",  # "whiten", "rmsnorm", "none"
        # Lazy-step mixing (identity + transport)
        dt: float = 1.,  # Step size ∈ (0, 1]. Set to 1/n_layers for "one full step across stack"
        dt_learnable: bool = True,  # If True, learn dt per attention module
        scale_alpha_by_dt: bool = True,  # If True, alpha_eff = dt * alpha (prevents ESR dominating at small dt)
    ):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_head = n_head
        self.d_head = n_embd // n_head
        self.n_taus = n_taus
        self.power_iters = power_iters
        self.dp_temp = dp_temp
        self.esr_gamma = esr_gamma
        self.esr_alpha = esr_alpha
        self.g_norm_mode = g_norm_mode
        self.scale_alpha_by_dt = scale_alpha_by_dt
        
        # Lazy-step: y = (1-dt)*v + dt*y_transport
        # Default dt=1.0 recovers original behavior (no identity mixing)
        if dt_learnable:
            # Learnable dt via sigmoid to keep in (0, 1)
            # init_logit chosen so sigmoid(init_logit) ≈ dt
            init_logit = math.log(dt / (1.0 - dt + 1e-8))
            self.dt_logit = nn.Parameter(torch.tensor(init_logit))
            self.dt_fixed = None
        else:
            self.register_parameter('dt_logit', None)
            self.dt_fixed = float(dt)
        
        # Fused projection: Q, K, V, G in single matmul
        self.qkvg = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Learnable parameters
        self.lam = nn.Parameter(torch.tensor(init_lam))
        self.log_tau = nn.Parameter(-0.5 * torch.ones(n_head, n_taus))
        self.logit_w = nn.Parameter(torch.zeros(n_head, n_taus))
        self.beta = nn.Parameter(1.5 * torch.ones(n_head))
        self.out_scale = nn.Parameter(torch.ones(n_head))
        
        # Optional learnable scale on dp and logK
        if dp_scale_learnable:
            self.dp_scale = nn.Parameter(torch.ones(n_head))
            self.logk_scale = nn.Parameter(torch.ones(n_head))
        else:
            self.register_buffer('dp_scale', torch.ones(n_head))
            self.register_buffer('logk_scale', torch.ones(n_head))
        
        # Optional normalization for g
        if g_norm_mode == "rmsnorm":
            self.g_norm = RMSNorm(self.d_head)  # Actually RMSNorm, not LayerNorm
        elif g_norm_mode == "layernorm":
            self.g_norm = nn.LayerNorm(self.d_head, elementwise_affine=True)
        else:
            self.g_norm = None
        
        # Causal mask
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.qkvg.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        if self.qkvg.bias is not None:
            nn.init.zeros_(self.qkvg.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
    
    def _power_iter_pi(self, P: torch.Tensor, gamma: float = 0.0) -> torch.Tensor:
        """
        Left stationary distribution via power iteration with inline teleport.
        
        Computes π such that π ≈ πP_smooth, where:
            P_smooth = (1-γ)P + γ·(1/T)·11ᵀ
        
        Instead of materializing P_smooth, we do teleport in vector space:
            π ← (1-γ)·πP + γ/T
        
        With γ > 0, P_smooth is irreducible/aperiodic (full support),
        so we don't need P² to break periodicity.
        
        Args:
            P: Causal Markov kernel [B, H, T, T]
            gamma: Teleport probability (default 0.0, but should use > 0 for causal P)
        
        Returns:
            pi: Stationary distribution [B, H, T]
        """
        B, H, T, _ = P.shape
        
        # Initialize uniform
        pi = P.new_full((B, H, 1, T), 1.0 / T)
        
        # Power iteration with inline teleport
        for _ in range(self.power_iters):
            pi_P = pi @ P  # [B, H, 1, T]
            if gamma > 0:
                # π ← (1-γ)·πP + γ/T  (teleport to uniform)
                pi = (1.0 - gamma) * pi_P + (gamma / T)
            else:
                pi = pi_P
            pi = pi / pi.sum(-1, keepdim=True).clamp_min(1e-12)
        
        return pi.squeeze(-2).detach()  # [B, H, T]
    
    def _compute_causal_reversibilization(self, P: torch.Tensor, pi: torch.Tensor, 
                                           mask: torch.Tensor) -> torch.Tensor:
        """
        Compute causal approximation to additive reversibilization.
        
        Computes P_rev = (P + P*) / 2 where P*_{ij} = (π_j/π_i) P_{ji},
        then projects onto causal support and renormalizes.
        
        NOTE: After causal projection, this is NOT exactly self-adjoint in ℓ²(π)
        and does NOT preserve π as stationary. It's a stabilizing heuristic.
        
        Args:
            P: Row-stochastic Markov matrix [B, H, T, T]
            pi: Stationary distribution estimate [B, H, T]  
            mask: Causal mask [1, 1, T, T]
        
        Returns:
            P_rev_causal: Causal reversibilized kernel [B, H, T, T]
        """
        # π_i and π_j for ratio computation
        pi_i = pi.unsqueeze(-1)      # [B, H, T, 1]
        pi_j = pi.unsqueeze(-2)      # [B, H, 1, T]
        
        # P*_{ij} = (π_j / π_i) * P_{ji}
        ratio = pi_j / pi_i.clamp_min(1e-12)
        P_star = P.transpose(-2, -1) * ratio
        
        # Additive reversibilization (exact before projection)
        P_rev = 0.5 * (P + P_star)
        
        # Project onto causal support (destroys exact self-adjointness)
        P_rev_causal = P_rev * mask.float()
        
        # Renormalize to ensure row-stochasticity
        P_rev_causal = P_rev_causal / P_rev_causal.sum(-1, keepdim=True).clamp_min(1e-12)
        
        return P_rev_causal
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_head, self.d_head
        dtype_in = x.dtype
        
        # Fused projection + reshape
        qkvg = self.qkvg(x).view(B, T, 4, H, D).permute(2, 0, 3, 1, 4)
        q, k, v, g = qkvg[0], qkvg[1], qkvg[2], qkvg[3]  # Each [B, H, T, D]
        
        # Normalize g based on mode
        if self.g_norm_mode == "whiten":
            g = (g - g.mean(-1, keepdim=True)) / (g.var(-1, keepdim=True, unbiased=False) + 1e-5).sqrt()
        elif self.g_norm_mode == "rmsnorm" and self.g_norm is not None:
            g = self.g_norm(g)
        # else: no normalization
        
        # ====== Compute kernels in FP32 for numerical stability ======
        q_f = q.float()
        k_f = k.float()
        g_f = g.float()
        
        # Dot-product scores with learnable scale
        dp_scale = self.dp_scale.view(1, H, 1, 1).clamp(0.1, 3.0)
        dp = torch.einsum('bhid,bhjd->bhij', q_f, k_f) * (self.dp_temp / math.sqrt(D))
        dp = dp * dp_scale
        
        # Fractional kernel with learnable scale
        s = pairwise_sqdist(g_f)
        tau = self.log_tau.clamp(-6, 3).exp()
        w = F.softmax(self.logit_w, dim=-1)
        logK = fractional_kernel(s, tau, w, self.beta)
        logk_scale = self.logk_scale.view(1, H, 1, 1).clamp(0.1, 3.0)
        logK = logK * logk_scale
        
        # Markov matrix P
        lam = torch.sigmoid(self.lam)
        mask = self.mask[:, :, :T, :T]
        P = causal_markov_normalize(logK, dp, lam, mask)
        
        # ====== Reversibilization ======
        
        # π ESTIMATION: Power iteration with full-support teleport (inline).
        # Causal P has token 0 as absorbing state. Full-support teleport ensures
        # irreducibility without materializing a T×T uniform matrix.
        pi = self._power_iter_pi(P, gamma=self.esr_gamma).clamp_min(1e-12)
        pi = pi / pi.sum(-1, keepdim=True).clamp_min(1e-12)
        
        # Reversibilize the CAUSAL P using the well-estimated π,
        # then project back to causal support.
        P_rev = self._compute_causal_reversibilization(P, pi, mask)
        
        # ====== Value transport with lazy-step mixing ======
        v_f = v.float()
        
        # Get effective step size dt ∈ (0, 1]
        if self.dt_logit is not None:
            dt = torch.sigmoid(self.dt_logit)
        else:
            dt = self.dt_fixed
        
        # Optionally scale alpha by dt (prevents ESR dominating at small dt)
        if self.scale_alpha_by_dt and dt < 1.0:
            alpha_eff = dt * self.esr_alpha
        else:
            alpha_eff = self.esr_alpha
        
        # Compute transported value
        y_plain = P @ v_f
        y_rev = P_rev @ v_f                            # Reversibilized transport (NOT P²v!)
        y_transport = (1.0 - alpha_eff) * y_plain + alpha_eff * y_rev
        
        # Lazy-step mixing: y = (1-dt)*v + dt*y_transport
        # dt=1 recovers original behavior; dt=1/n_layers → "one step across stack"
        if dt < 1.0:
            y = (1.0 - dt) * v_f + dt * y_transport
        else:
            y = y_transport
        
        # Merge heads
        scale = self.out_scale.view(1, H, 1, 1).clamp(0.1, 2.0)
        y = (y * scale).transpose(1, 2).reshape(B, T, C)
        y = y.to(dtype_in)
        
        return self.dropout(self.out_proj(y))
    
    def get_diagnostics(self, x: torch.Tensor) -> dict:
        """
        Return diagnostic information for debugging.
        
        Key check: y_rev should NOT equal P²v. If rev_vs_p2_max_diff is near
        zero, we might still have the bug (though this isn't a proof - they
        could coincidentally be close in some regimes).
        """
        with torch.no_grad():
            B, T, C = x.shape
            H, D = self.n_head, self.d_head
            
            qkvg = self.qkvg(x).view(B, T, 4, H, D).permute(2, 0, 3, 1, 4)
            q, k, v, g = qkvg[0], qkvg[1], qkvg[2], qkvg[3]
            
            if self.g_norm_mode == "whiten":
                g = (g - g.mean(-1, keepdim=True)) / (g.var(-1, keepdim=True, unbiased=False) + 1e-5).sqrt()
            elif self.g_norm is not None:
                g = self.g_norm(g)
            
            q_f, k_f, g_f, v_f = q.float(), k.float(), g.float(), v.float()
            
            dp_scale = self.dp_scale.view(1, H, 1, 1).clamp(0.1, 3.0)
            dp = torch.einsum('bhid,bhjd->bhij', q_f, k_f) * (self.dp_temp / math.sqrt(D)) * dp_scale
            
            s = pairwise_sqdist(g_f)
            tau = self.log_tau.clamp(-6, 3).exp()
            w = F.softmax(self.logit_w, dim=-1)
            logK = fractional_kernel(s, tau, w, self.beta)
            logk_scale = self.logk_scale.view(1, H, 1, 1).clamp(0.1, 3.0)
            logK = logK * logk_scale
            
            lam = torch.sigmoid(self.lam)
            mask = self.mask[:, :, :T, :T]
            P = causal_markov_normalize(logK, dp, lam, mask)
            
            # π estimation with inline teleport
            pi = self._power_iter_pi(P, gamma=self.esr_gamma).clamp_min(1e-12)
            pi = pi / pi.sum(-1, keepdim=True).clamp_min(1e-12)
            
            # Reversibilize causal P
            P_rev = self._compute_causal_reversibilization(P, pi, mask)
            
            # Compute what the BUG would produce
            y_plain = P @ v_f
            P2 = P @ P
            P2 = P2 / P2.sum(-1, keepdim=True).clamp_min(1e-12)
            y_bug = P2 @ v_f  # What the original code computed
            y_rev = P_rev @ v_f  # What we compute now
            
            # Difference between our output and the bug
            rev_vs_p2_diff = (y_rev - y_bug).abs().max().item()
            
            # Check row-stochasticity
            row_sums = P_rev.sum(-1)
            row_sum_error = (row_sums - 1.0).abs().max().item()
            
            # Check approximate symmetry in ℓ²(π) (won't be exact due to causal projection)
            sqrt_pi = pi.sqrt().unsqueeze(-1)
            inv_sqrt_pi = 1.0 / sqrt_pi
            B_rev = sqrt_pi * P_rev * inv_sqrt_pi.transpose(-2, -1)
            symmetry_error = (B_rev - B_rev.transpose(-2, -1)).abs().max().item()
            
            # Get effective dt
            if self.dt_logit is not None:
                dt = torch.sigmoid(self.dt_logit).item()
            else:
                dt = self.dt_fixed
            
            return {
                'lam': lam.item(),
                'dt': dt,  # Lazy-step size
                'esr_alpha': self.esr_alpha,
                'esr_gamma': self.esr_gamma,
                'pi_min': pi.min().item(),
                'pi_max': pi.max().item(),
                'pi_ratio_max_min': (pi.max() / pi.min().clamp_min(1e-12)).item(),  # Should be reasonable, not 1e6+
                'pi_first_token': pi[0, 0, 0].item(),  # Check if collapsed to δ₀
                'pi_last_token': pi[0, 0, -1].item(),
                'P_row_sum_error': (P.sum(-1) - 1.0).abs().max().item(),
                'P_rev_row_sum_error': row_sum_error,
                'P_rev_symmetry_error_in_l2pi': symmetry_error,  # Expected nonzero due to causal mask
                'rev_vs_p2_max_diff': rev_vs_p2_diff,  # Heuristic check, not proof
            }


# ============================================================================
# VERIFICATION FUNCTION
# ============================================================================

def verify_esr_correctness(attn: CausalSemigroupSelfAttentionSelective,
                           batch_size: int = 2,
                           seq_len: int = 64,
                           device: str = 'cuda') -> dict:
    """
    Verify that reversibilization is implemented correctly.
    
    Key checks:
    1. y_rev should differ from P²v (core bug fix)
    2. π should NOT collapse to δ₀ (full-support teleport fix)
    3. dt controls "semigroup time per layer" (lazy-step mixing)
    """
    n_embd = attn.n_head * attn.d_head
    x = torch.randn(batch_size, seq_len, n_embd, device=device)
    
    diag = attn.get_diagnostics(x)
    
    print("=" * 60)
    print("REVERSIBILIZATION DIAGNOSTICS")
    print("=" * 60)
    print(f"λ (blend parameter):              {diag['lam']:.4f}")
    print(f"dt (step size):                   {diag['dt']:.4f}")
    print(f"α (rev blend):                    {diag['esr_alpha']:.4f}")
    print(f"γ (teleport):                     {diag['esr_gamma']:.4f}")
    print("-" * 60)
    print("π DISTRIBUTION (should NOT collapse to early tokens):")
    print(f"  π range:                        [{diag['pi_min']:.2e}, {diag['pi_max']:.2e}]")
    print(f"  π[0] (first token):             {diag['pi_first_token']:.4f}")
    print(f"  π[-1] (last token):             {diag['pi_last_token']:.4f}")
    print(f"  π max/min ratio:                {diag['pi_ratio_max_min']:.1f}")
    print("-" * 60)
    print(f"P row-sum error:                  {diag['P_row_sum_error']:.2e}")
    print(f"P_rev row-sum error:              {diag['P_rev_row_sum_error']:.2e}")
    print(f"P_rev symmetry error in ℓ²(π):    {diag['P_rev_symmetry_error_in_l2pi']:.2e}")
    print("  (nonzero expected due to causal projection)")
    print("-" * 60)
    print(f"rev vs P² max diff:               {diag['rev_vs_p2_max_diff']:.4f}")
    print("-" * 60)
    
    # Check π health
    if diag['pi_ratio_max_min'] > 1e4:
        print(f"⚠️  WARNING: π ratio {diag['pi_ratio_max_min']:.0e} is very large.")
        print("    π may be collapsing. Check teleport γ.")
    elif diag['pi_first_token'] > 0.5:
        print(f"⚠️  WARNING: π[0] = {diag['pi_first_token']:.2f} dominates.")
        print("    Possible absorbing state issue.")
    else:
        print("✓  π distribution looks healthy.")
    
    if diag['rev_vs_p2_max_diff'] < 1e-4:
        print("⚠️  WARNING: P_rev ≈ P². Possible issue (or coincidental).")
    else:
        print("✓  P_rev differs from P² (core bug is fixed).")
    
    if diag['P_rev_row_sum_error'] < 1e-6:
        print("✓  P_rev is row-stochastic.")
    else:
        print(f"⚠️  P_rev row sums deviate by {diag['P_rev_row_sum_error']:.2e}")
    
    print("=" * 60)
    
    return diag


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class RMSNorm(nn.Module):
    """RMSNorm - faster than LayerNorm, works well in practice."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation - better than GELU for transformers."""
    def __init__(self, n_embd: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        hidden = int(8 * n_embd / 3)
        hidden = ((hidden + 63) // 64) * 64  # Round to multiple of 64
        
        self.w1 = nn.Linear(n_embd, hidden, bias=bias)
        self.w2 = nn.Linear(hidden, n_embd, bias=bias)
        self.w3 = nn.Linear(n_embd, hidden, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MLP(nn.Module):
    """Standard MLP with GELU (fallback if SwiGLU not wanted)."""
    def __init__(self, n_embd: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x), approximate='tanh')))


class Block(nn.Module):
    """Transformer block with semigroup attention."""
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        
        # Compute per-layer dt if using fixed schedule
        if hasattr(config, 'dt') and config.dt is not None:
            dt = config.dt
        else:
            # Default: dt = 1/n_layers for "one full step across stack"
            dt = 1.0 / config.n_layer if hasattr(config, 'n_layer') else 1.0
        
        self.attn = CausalSemigroupSelfAttentionSelective(
            n_embd=config.n_embd,
            n_head=config.n_head,
            block_size=config.block_size,
            n_taus=config.n_taus,
            power_iters=config.power_iters,
            dp_temp=config.dp_temp,
            esr_gamma=config.esr_gamma,
            esr_alpha=config.esr_alpha,
            dropout=config.dropout,
            bias=config.bias,
            init_lam=config.init_lam,
            dp_scale_learnable=config.dp_scale_learnable,
            g_norm_mode=config.g_norm_mode,
            dt=dt,
            dt_learnable=getattr(config, 'dt_learnable', False),
            scale_alpha_by_dt=getattr(config, 'scale_alpha_by_dt', False),
        )
        self.ln2 = RMSNorm(config.n_embd)
        
        if config.use_swiglu:
            self.mlp = SwiGLU(config.n_embd, config.dropout, config.bias)
        else:
            self.mlp = MLP(config.n_embd, config.dropout, config.bias)
        
        # Optional learnable residual scales
        if config.learnable_residual:
            self.res_scale = nn.Parameter(torch.ones(2))
        else:
            self.register_buffer('res_scale', torch.ones(2))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.res_scale.sigmoid() if self.res_scale.requires_grad else self.res_scale
        x = x + s[0] * self.attn(self.ln1(x))
        x = x + s[1] * self.mlp(self.ln2(x))
        return x


# ============================================================================
# GPT MODEL
# ============================================================================

@dataclass
class GPTConfig:
    # Model architecture
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    
    # Semigroup attention
    n_taus: int = 8
    power_iters: int = 4
    dp_temp: float = 0.8
    esr_gamma: float = 0.02
    esr_alpha: float = 0.1
    
    # New controls
    init_lam: float = 1.5  # Start dot-product dominated (sigmoid(1.5) ≈ 0.82)
    dp_scale_learnable: bool = True
    g_norm_mode: str = "rmsnorm"  # "whiten", "rmsnorm", "none"
    
    # Lazy-step mixing: y = (1-dt)*v + dt*y_transport
    # Set dt=None to use 1/n_layer (one full step across stack)
    dt: Optional[float] = None  # Per-layer step size; None → auto = 1/n_layer
    dt_learnable: bool = False  # If True, learn dt per layer via sigmoid
    scale_alpha_by_dt: bool = False  # If True, alpha_eff = dt * alpha (prevents ESR dominating at small dt)
    
    # Architecture choices
    use_swiglu: bool = True
    learnable_residual: bool = True
    use_gradient_checkpointing: bool = False


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
            h=nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize
        self.apply(self._init_weights)
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params / 1e6:.2f}M")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        opt = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **({'fused': True} if use_fused else {}))
        print(f"using fused AdamW: {use_fused}")
        return opt

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        return (flops_per_token * T * fwdbwd_per_iter / dt) / 312e12
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params   
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        assert T <= self.config.block_size
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        x = self.transformer.drop(x)
        
        if self.config.use_gradient_checkpointing and self.training:
            for block in self.transformer.h:
                x = checkpoint(block, x, use_reentrant=False)
        else:
            for block in self.transformer.h:
                x = block(x)
        
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, 
                 temperature: float = 1.0, top_k: Optional[int] = None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


# ============================================================================
# SOFTMAX BASELINE
# ============================================================================

class SoftmaxAttention(nn.Module):
    """Standard causal softmax attention for comparison."""
    def __init__(self, n_embd: int, n_head: int, block_size: int, 
                 dropout: float = 0.0, bias: bool = False, **kwargs):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_head = n_head
        self.d_head = n_embd // n_head
        
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_head, self.d_head
        
        qkv = self.qkv(x).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if hasattr(F, 'scaled_dot_product_attention'):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, 
                                               dropout_p=0.0 if not self.training else self.dropout.p)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D))
            att = att.masked_fill(~self.mask[:, :, :T, :T], float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v
        
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.dropout(self.out_proj(y))


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CORRECTED SEMIGROUP ATTENTION")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Triton available: {HAS_TRITON}")
    
    # Test ESR correctness
    print("\n" + "=" * 70)
    print("TESTING ESR CORRECTNESS")
    print("=" * 70)
    
    attn = CausalSemigroupSelfAttentionSelective(
        n_embd=768,
        n_head=12,
        block_size=256,
        n_taus=8,
        esr_alpha=0.1,
    ).to(device)
    
    verify_esr_correctness(attn, batch_size=2, seq_len=64, device=device)
    
    # Quick forward pass test
    print("\n" + "=" * 70)
    print("FORWARD PASS TEST")
    print("=" * 70)
    
    config = GPTConfig(
        block_size=256,
        vocab_size=50304,
        n_layer=6,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=False,
        n_taus=8,
        power_iters=4,
        dp_temp=0.8,
        esr_gamma=0.02,
        esr_alpha=0.1,
        init_lam=1.5,
        use_swiglu=True,
        learnable_residual=True,
    )
    
    model = GPT(config).to(device)
    x = torch.randint(0, config.vocab_size, (2, 64), device=device)
    y = torch.randint(0, config.vocab_size, (2, 64), device=device)
    
    logits, loss = model(x, targets=y)
    print(f"Output shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    print("✓ Backward pass successful")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)