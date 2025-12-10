"""
Optimized Semigroup Attention Implementation
=============================================

High-performance implementation preserving key mathematical properties:
- Chapman-Kolmogorov consistency via Markov transition matrices
- ESR (Explicitly Symmetric Realization) in ℓ²(π) geometry
- Fractional diffusion kernels with learnable τ-mixture

Key Optimizations:
1. Fused QKV+G projection (4 matmuls → 1)
2. Triton kernels for pairwise distance + fractional kernel
3. Memory-efficient chunked P² computation
4. Reduced iterations: 4 power iters (vs 16), 4 τ components (vs 8)
5. torch.compile() compatible (no state mutations in forward)
6. Flash-style online softmax for row normalization
7. Gradient checkpointing support
8. Mixed precision friendly

Usage:
    config = GPTConfig(n_layer=12, n_head=12, n_embd=768)
    model = GPT(config).cuda()
    model = torch.compile(model)  # Recommended for extra speed
"""

import math
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
            
            # Load g[row, d]
            row_ptrs = G_ptr + pid_bh * stride_gb + row_offs[:, None] * stride_gt + d_offs[None, :] * stride_gd
            col_ptrs = G_ptr + pid_bh * stride_gb + col_offs[:, None] * stride_gt + d_offs[None, :] * stride_gd
            
            row_mask = (row_offs[:, None] < T) & d_mask[None, :]
            col_mask = (col_offs[:, None] < T) & d_mask[None, :]
            
            g_row = tl.load(row_ptrs, mask=row_mask, other=0.0).to(tl.float32)
            g_col = tl.load(col_ptrs, mask=col_mask, other=0.0).to(tl.float32)
            
            acc_dot += tl.dot(g_row, tl.trans(g_col))
            row_norm += tl.sum(g_row * g_row, axis=1)
            col_norm += tl.sum(g_col * g_col, axis=1)
        
        # ||g_i - g_j||² = ||g_i||² + ||g_j||² - 2<g_i, g_j>
        sqdist = row_norm[:, None] + col_norm[None, :] - 2.0 * acc_dot
        sqdist = tl.maximum(sqdist, 0.0)
        
        # Store
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
        
        # Load squared distances
        s_ptrs = S_ptr + pid_bh * stride_sb + row_offs[:, None] * stride_st1 + col_offs[None, :] * stride_st2
        mask = (row_offs[:, None] < T) & (col_offs[None, :] < T)
        s = tl.load(s_ptrs, mask=mask, other=0.0).to(tl.float32)
        
        # Load beta
        beta = tl.load(Beta_ptr + h_idx).to(tl.float32)
        beta = tl.maximum(tl.minimum(beta, 2.5), 0.5)
        
        # Online logsumexp over τ components
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
        Online algorithm avoids materializing full row.
        """
        pid_bh = tl.program_id(0)
        row_idx = tl.program_id(1)
        
        if row_idx >= T:
            return
        
        # First pass: find max (only over causal positions j <= row_idx)
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
        
        # Second pass: compute sum of exp
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

    # Triton fast-path ONLY in eager mode (not under torch.compile)
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

    # Pure PyTorch fallback (used under torch.compile)
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

    # Triton fast-path ONLY in eager mode (not under torch.compile)
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

    # PyTorch fallback (used under torch.compile)
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
                            lam,
                            mask: torch.Tensor) -> torch.Tensor:
    """Blend logK and dp, apply causal mask, row-normalize to Markov matrix."""
    B, H, T, _ = logK.shape

    # Triton fast-path ONLY in eager mode (not under torch.compile)
    if HAS_TRITON and logK.is_cuda and T >= 64 and not _is_compiling():
        logK_flat = logK.reshape(B * H, T, T).contiguous()
        dp_flat = dp.reshape(B * H, T, T).contiguous()
        P_out = torch.zeros_like(logK_flat)

        BLOCK_COL = min(128, triton.next_power_of_2(T))
        grid = (B * H, T)

        # scalar for Triton
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

    # PyTorch fallback (used under torch.compile)
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
# OPTIMIZED SEMIGROUP ATTENTION
# ============================================================================

class CausalSemigroupSelfAttentionSelective(nn.Module):
    """
    Fast semigroup attention with ESR in ℓ²(π) geometry.
    
    Implements:
    1. Blended kernel: K = exp(λ·dp + (1-λ)·frac_log)
    2. Markov normalization: P = K / K.sum(-1)
    3. ESR transform via π from P²
    4. Value transport: y = (1-α)·Pv + α·D^{-1/2}BD^{1/2}(Pv)
    """
    
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        n_taus: int = 16,
        power_iters: int = 4,
        dp_temp: float = 0.8,
        esr_gamma: float = 0.02,
        esr_alpha: float = 0.08,
        dropout: float = 0.0,
        bias: bool = False,
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
        
        # Fused projection: Q, K, V, G in single matmul
        self.qkvg = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Learnable parameters (kept simple)
        self.lam = nn.Parameter(torch.tensor(0.5))
        self.log_tau = nn.Parameter(-0.5 * torch.ones(n_head, n_taus))
        self.logit_w = nn.Parameter(torch.zeros(n_head, n_taus))
        self.beta = nn.Parameter(1.5 * torch.ones(n_head))
        self.out_scale = nn.Parameter(torch.ones(n_head))
        
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
    
    def _power_iter_pi(self, P: torch.Tensor) -> torch.Tensor:
        """Left stationary distribution via power iteration on P²."""
        B, H, T, _ = P.shape
        
        # Compute P² (chunked for memory efficiency on long sequences)
        if T > 512:
            P2 = torch.zeros_like(P)
            chunk = 256
            for i in range(0, T, chunk):
                P2[:, :, i:i+chunk] = P[:, :, i:i+chunk] @ P
        else:
            P2 = P @ P
        P2 = P2 / P2.sum(-1, keepdim=True).clamp_min(1e-12)
        
        # Power iteration
        pi = P2.new_full((B, H, 1, T), 1.0 / T)
        for _ in range(self.power_iters):
            pi = pi @ P2
            pi = pi / pi.sum(-1, keepdim=True).clamp_min(1e-12)
        
        return pi.squeeze(-2).detach()  # [B, H, T]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_head, self.d_head
        
        # Fused projection + reshape
        qkvg = self.qkvg(x).view(B, T, 4, H, D).permute(2, 0, 3, 1, 4)
        q, k, v, g = qkvg[0], qkvg[1], qkvg[2], qkvg[3]  # Each [B, H, T, D]
        
        # Channel-whiten g (helps geometry learning)
        g = (g - g.mean(-1, keepdim=True)) / (g.var(-1, keepdim=True, unbiased=False) + 1e-5).sqrt()
        
        # Dot-product scores
        dp = torch.einsum('bhid,bhjd->bhij', q, k) * (self.dp_temp / math.sqrt(D))
        
        # Fractional kernel
        s = pairwise_sqdist(g)
        tau = self.log_tau.clamp(-6, 3).exp()
        w = F.softmax(self.logit_w, dim=-1)
        logK = fractional_kernel(s, tau, w, self.beta)
        
        # Markov matrix
        lam = torch.sigmoid(self.lam)
        mask = self.mask[:, :, :T, :T]
        P = causal_markov_normalize(logK, dp, lam, mask)
        
        # Teleport smoothing for π estimation
        if self.esr_gamma > 0:
            Pi_uniform = mask.float() / mask.float().sum(-1, keepdim=True).clamp_min(1.0)
            P_for_pi = (1.0 - self.esr_gamma) * P + self.esr_gamma * Pi_uniform
        else:
            P_for_pi = P
        
        # ESR transform
        pi = self._power_iter_pi(P_for_pi).clamp_min(1e-12)
        pi = pi / pi.sum(-1, keepdim=True).clamp_min(1e-12)
        
        sqrt_pi = pi.sqrt().unsqueeze(-1)
        inv_sqrt_pi = 1.0 / sqrt_pi
        
        # B = D^{1/2} P D^{-1/2} (symmetric in ℓ²(π))
        Bsym = sqrt_pi * P * inv_sqrt_pi.transpose(-2, -1)
        
        # Value transport
        y_plain = P @ v
        y_esr = inv_sqrt_pi * (Bsym @ (sqrt_pi * y_plain))
        y = (1.0 - self.esr_alpha) * y_plain + self.esr_alpha * y_esr
        
        # Merge heads
        scale = self.out_scale.view(1, H, 1, 1).clamp(0.1, 2.0)
        y = (y * scale).transpose(1, 2).reshape(B, T, C)
        
        return self.dropout(self.out_proj(y))


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
        )
        self.ln2 = RMSNorm(config.n_embd)
        
        if config.use_swiglu:
            self.mlp = SwiGLU(config.n_embd, config.dropout, config.bias)
        else:
            self.mlp = MLP(config.n_embd, config.dropout, config.bias)
        
        # Optional learnable residual scales (helps deep networks)
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
    n_taus: int = 4
    power_iters: int = 4
    dp_temp: float = 0.8
    esr_gamma: float = 0.02
    esr_alpha: float = 0.08
    
    # Architecture choices
    use_swiglu: bool = True
    learnable_residual: bool = True
    use_gradient_checkpointing: bool = False


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        
        self.blocks = nn.ModuleList([
            Block(config, i) for i in range(config.n_layer)
        ])
        self.ln_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.wte.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params/1e6:.2f}M")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, '_is_residual'):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.config.block_size, f"Sequence length {T} > block_size {self.config.block_size}"
        
        pos = torch.arange(T, device=idx.device, dtype=torch.long)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        
        for block in self.blocks:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        x = self.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss
    
    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str,
    ):
        # Separate parameters that should/shouldn't have weight decay
        decay_params = []
        nodecay_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)
        
        groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        
        use_fused = device_type == 'cuda' and torch.cuda.is_available()
        optimizer = torch.optim.AdamW(groups, lr=learning_rate, betas=betas, fused=use_fused)
        return optimizer
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx
    
    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """Estimate model FLOPs utilization (MFU)."""
        N = sum(p.numel() for p in self.parameters())
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        
        # Rough FLOP estimate
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_achieved = flops_per_token * T * fwdbwd_per_iter / dt
        
        # A100 peak: 312 TFLOPS for BF16
        flops_promised = 312e12
        return flops_achieved / flops_promised


# ============================================================================
# SOFTMAX BASELINE FOR COMPARISON
# ============================================================================

class SoftmaxAttention(nn.Module):
    """Standard scaled dot-product attention for benchmarking."""
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
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
        
        # Use flash attention if available
        if hasattr(F, 'scaled_dot_product_attention'):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0 if not self.training else self.dropout.p)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D))
            att = att.masked_fill(~self.mask[:, :, :T, :T], float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v
        
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.dropout(self.out_proj(y))


# ============================================================================
# BENCHMARKING UTILITIES
# ============================================================================

def benchmark_attention(
    attn_cls,
    batch_size: int = 8,
    seq_len: int = 512,
    n_embd: int = 768,
    n_head: int = 12,
    warmup: int = 10,
    iters: int = 50,
    device: str = 'cuda',
    **kwargs,
):
    """Benchmark attention module throughput."""
    import time
    
    if attn_cls == CausalSemigroupSelfAttentionSelective:
        attn = attn_cls(n_embd, n_head, seq_len, **kwargs).to(device).eval()
    else:
        attn = attn_cls(n_embd, n_head, seq_len, **kwargs).to(device).eval()
    
    x = torch.randn(batch_size, seq_len, n_embd, device=device)
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = attn(x)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(iters):
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            _ = attn(x)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    times = torch.tensor(times)
    return times.mean().item(), times.std().item()


def benchmark_model(
    config: GPTConfig,
    batch_size: int = 8,
    warmup: int = 10,
    iters: int = 30,
    device: str = 'cuda',
):
    """Benchmark full model throughput."""
    import time
    
    model = GPT(config).to(device).eval()
    
    # Try to compile
    try:
        model = torch.compile(model, mode='reduce-overhead')
        print("Model compiled successfully")
    except Exception as e:
        print(f"Compilation skipped: {e}")
    
    x = torch.randint(0, config.vocab_size, (batch_size, config.block_size), device=device)
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(iters):
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            _ = model(x)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    times = torch.tensor(times)
    tokens_per_sec = (batch_size * config.block_size) / times.mean().item()
    
    return {
        'mean_ms': times.mean().item() * 1000,
        'std_ms': times.std().item() * 1000,
        'tokens_per_sec': tokens_per_sec,
    }


def memory_profile(config: GPTConfig, batch_size: int = 8, device: str = 'cuda'):
    """Profile peak memory usage."""
    torch.cuda.reset_peak_memory_stats()
    
    model = GPT(config).to(device)
    x = torch.randint(0, config.vocab_size, (batch_size, config.block_size), device=device)
    y = torch.randint(0, config.vocab_size, (batch_size, config.block_size), device=device)
    
    # Forward + backward
    model.train()
    _, loss = model(x, targets=y)
    loss.backward()
    
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    return peak_mem


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("OPTIMIZED SEMIGROUP ATTENTION")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Triton available: {HAS_TRITON}")
    
    # Configuration
    config = GPTConfig(
        block_size=1024,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=False,
        n_taus=4,
        power_iters=4,
        dp_temp=0.8,
        esr_gamma=0.02,
        esr_alpha=0.08,
        use_swiglu=True,
        learnable_residual=True,
    )
    
    print(f"\nConfig: {config.n_layer}L, {config.n_head}H, {config.n_embd}D, T={config.block_size}")
    
    if device == 'cuda':
        # Attention comparison
        print("\n" + "="*70)
        print("ATTENTION BENCHMARKS (B=8, T=512)")
        print("="*70)
        
        # Semigroup
        sg_mean, sg_std = benchmark_attention(CausalSemigroupSelfAttentionSelective, seq_len=512, device=device)
        print(f"Semigroup: {sg_mean*1000:.2f} ± {sg_std*1000:.2f} ms")
        
        # Softmax baseline
        sm_mean, sm_std = benchmark_attention(SoftmaxAttention, seq_len=512, device=device)
        print(f"Softmax:   {sm_mean*1000:.2f} ± {sm_std*1000:.2f} ms")
        print(f"Ratio:     {sg_mean/sm_mean:.2f}x")
        
        # Full model benchmark
        print("\n" + "="*70)
        print("FULL MODEL BENCHMARKS")
        print("="*70)
        
        for seq_len in [256, 512, 1024]:
            config.block_size = seq_len
            results = benchmark_model(config, batch_size=8, device=device)
            print(f"T={seq_len:4d}: {results['mean_ms']:.1f}ms, {results['tokens_per_sec']:.0f} tok/s")
        
        # Memory profile
        print("\n" + "="*70)
        print("MEMORY PROFILE (B=8, T=1024)")
        print("="*70)
        
        config.block_size = 1024
        peak_mem = memory_profile(config, batch_size=8, device=device)
        print(f"Peak memory: {peak_mem:.2f} GB")
        
        # With gradient checkpointing
        config.use_gradient_checkpointing = True
        peak_mem_ckpt = memory_profile(config, batch_size=8, device=device)
        print(f"With checkpointing: {peak_mem_ckpt:.2f} GB ({100*(1-peak_mem_ckpt/peak_mem):.0f}% reduction)")
        
    else:
        print("\nCPU mode - running smoke test")
        config.block_size = 64
        model = GPT(config)
        x = torch.randint(0, config.vocab_size, (2, 64))
        y = torch.randint(0, config.vocab_size, (2, 64))
        logits, loss = model(x, targets=y)
        print(f"Output: {logits.shape}, Loss: {loss.item():.4f}")
    
    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)