# Semigroup Attention Integration in Megatron-LM

## Overview

Details for integrating  **semigroup attention** into Megatron-LM's GPT training pipeline on the Clariden cluster.

---

## Current steps

### 1. Semigroup Attention Wrapper

**File:** `megatron/core/transformer/semigroup_attention.py`

 Megatron-compatible wrapper around the standalone `CausalSemigroupSelfAttentionSelective` implementation from George's `semigroup_fast.py`.

**Key features:**
- Inherits from `MegatronModule`
- Handles tensor shape transposition between Megatron's `[S, B, H]` and semigroup's `[B, S, H]` format
- Extracts config parameters (hidden_size, num_heads, max_position_embeddings) from Megatron's `TransformerConfig`
- Currently requires tensor-parallel size = 1 (I tested on single-GPU in general i have problem with testing the containers and i dont have permission for what they had on the github and for the bad conenction it was messy to test that)

### 2. Created GPT Layer Spec for Semigroup Attention

**File:** `megatron/core/models/gpt/gpt_layer_specs.py`

Added function `get_gpt_layer_semigroup_spec()` that returns a `ModuleSpec` using:
- `SemigroupSelfAttentionMegatron` for self-attention
- Standard MLP from Megatron

### 3. Fixed Spec Loading in pretrain_gpt.py

**File:** `pretrain_gpt.py`

The `--spec` argument now loads and invokes custom spec functions:
```python
_spec_obj = import_module(args.spec)
transformer_layer_spec = _spec_obj() if callable(_spec_obj) else _spec_obj
```

### 5. Due to not be able to load the container that they.had I added single-gpu bypasses to do an end-to-end test for megatron integration 

**Files modified (check for  all the comments with the tag `[SEMIGROUP]`):**
- `megatron/training/initialize.py` — Uses gloo not nccl backend for world_size=1
- `megatron/training/training.py` — Skips DDP wrapper and `zero_grad_buffer` for world_size=1
- `megatron/core/distributed/finalize_model_grads.py` — Skips gradient all-reduce for world_size=1

**Note:** Btw the bypasses only activate when `world_size == 1`, so i believe that for multi-GPU runs using the normal NCCL/DDP path for the correct container it will work.

---

## for running just:

```bash
sbatch run_gpt_semigroup.sh
```

It will run a tiny 2-layer model with the new attention for 10 iterations to validate the integration.

### Some specs though:

In `run_gpt_semigroup.sh`:
```bash
--spec megatron.core.models.gpt.gpt_layer_specs get_gpt_layer_semigroup_spec  # Semigroup attention
--transformer-impl local      # Don't use TransformerEngine
--mock-data                   # Data are synthetic we will sub all the paths with the correct ones for what imanol said
--tokenizer-type NullTokenizer
```


## Validation Results
check the file gpt-semi-1215889.out at the bottom it hass an example output
