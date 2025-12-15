from typing import Optional, Tuple

import torch
from torch import Tensor

from megatron.core import parallel_state
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

from semigroup_fast import CausalSemigroupSelfAttentionSelective


class SemigroupSelfAttentionMegatron(MegatronModule):
    """Wrapper for plugging `CausalSemigroupSelfAttentionSelective` into Megatron.

    This module deliberately ignores Megatron's attention mask and rotary/KV
    cache machinery and relies on the internal causal mask in the semigroup
    implementation. It is intended for small-scale experiments with
    `tensor_model_parallel_size == 1` and no inference KV cache.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        cp_comm_type: Optional[str] = None,  # unused, kept for API symmetry
    ) -> None:
        super().__init__(config=config)

        assert (
            parallel_state.get_tensor_model_parallel_world_size() == 1
        ), "SemigroupSelfAttentionMegatron currently supports only TP=1"

        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type

        # Determine maximum supported sequence length for the causal mask.
        # `TransformerConfig` (GPT) does not expose max position directly, but
        # `MLATransformerConfig` does via `max_position_embeddings`. To keep
        # this wrapper generic and robust, prefer that attribute when it
        # exists and fall back to a conservative value (sufficient for our
        # small smoke tests which use seq_length=128).
        block_size = getattr(config, "max_position_embeddings", 2048)

        # Map Megatron config to semigroup attention hyperparameters.
        self.semigroup_attn = CausalSemigroupSelfAttentionSelective(
            n_embd=config.hidden_size,
            n_head=config.num_attention_heads,
            block_size=block_size,
            n_taus=4,
            power_iters=4,
            dp_temp=0.8,
            esr_gamma=0.02,
            esr_alpha=0.08,
            dropout=float(config.hidden_dropout),
            bias=False,
        )

        if parallel_state.get_tensor_model_parallel_rank() == 0:
            print(
                f"[SemigroupSelfAttentionMegatron] Initialized on layer {layer_number} "
                f"with hidden_size={config.hidden_size}, num_heads={config.num_attention_heads}, "
                f"max_seq={block_size}"
            )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        key_value_states: Optional[Tensor] = None,
        inference_params=None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params=None,
        sequence_len_offset: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Megatron-style forward.

        Args:
            hidden_states: [seq, batch, hidden]

        Returns:
            output: [seq, batch, hidden]
            bias: always None (no fused bias path here)
        """

        # Semigroup attention is causal internally; we only need to ensure
        # the sequence length does not exceed its configured block size.
        seq_len, batch_size, hidden_size = hidden_states.shape
        assert (
            seq_len <= self.semigroup_attn.mask.size(-1)
        ), "Sequence length exceeds semigroup block_size"

        # Convert from [S, B, H] -> [B, S, H]
        x = hidden_states.transpose(0, 1).contiguous()

        # Run semigroup attention, ignore Megatron's attention_mask for now.
        y = self.semigroup_attn(x)

        # Back to [S, B, H]
        y = y.transpose(0, 1).contiguous()

        return y, None
