# KDA (Kimi Delta Attention) — NPU Triton Implementation

> Migrated from [Theta/flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention) (`hpu_dev` branch)  
> Target: `fla/ops/triton/triton_core/kda/`  
> Architecture: **100% Triton**, no AscendC replacement

---

## 1. Overview

KDA (Kimi Delta Attention) is a linear attention variant developed by Moonshot AI. It extends Gated Delta Attention with a learnable gate mechanism (`A_log + dt_bias + softplus`), distinguishing it from simpler GDN decay schemes.

This module provides the **NPU-compatible Triton-only** implementation of KDA, with all kernels written in Triton and validated to run on Ascend 910B via the triton-ascend backend. No AscendC C++ custom operators are used.

---

## 2. File Structure

```
kda/
├── __init__.py                      # Public API exports
├── chunk.py                         # Main entry point: ChunkKDAFunction + chunk_kda()
├── chunk_fwd.py                     # Chunk forward orchestration
├── chunk_bwd.py                     # Chunk backward kernels + orchestration
├── chunk_delta_h.py                 # Hidden state forward/backward recurrence (extracted from GDN)
├── chunk_fwd_o.py                   # Output projection kernel (extracted from GLA)
├── chunk_intra.py                   # KDA intra-chunk attention kernels (Aqk + Akk)
├── chunk_intra_token_parallel.py    # Token-parallel diagonal block kernel variant
├── fused_recurrent.py               # Fused recurrent decode kernel (for autoregressive inference)
├── gate.py                          # KDA gate computation: fused_kda_gate(), kda_gate_chunk_cumsum()
├── naive.py                         # Pure PyTorch reference implementations (naive_recurrent_kda, naive_chunk_kda)
├── wy_fast.py                       # KDA WY representation recomputation kernel (with g_kk decay)
└── README.md                        # This file
```

### Source Origin

| File | Origin | Type |
|------|--------|------|
| `chunk.py` | `fla/ops/kda/chunk.py` | KDA-specific |
| `chunk_fwd.py` | `fla/ops/kda/chunk_fwd.py` | KDA-specific |
| `chunk_bwd.py` | `fla/ops/kda/chunk_bwd.py` | KDA-specific |
| `chunk_intra.py` | `fla/ops/kda/chunk_intra.py` | KDA-specific |
| `chunk_intra_token_parallel.py` | `fla/ops/kda/chunk_intra_token_parallel.py` | KDA-specific |
| `gate.py` | `fla/ops/kda/gate.py` | KDA-specific |
| `wy_fast.py` | `fla/ops/kda/wy_fast.py` | KDA-specific |
| `fused_recurrent.py` | `fla/ops/kda/fused_recurrent.py` | KDA-specific |
| `naive.py` | `fla/ops/kda/naive.py` | KDA-specific |
| `chunk_delta_h.py` | `fla/ops/common/chunk_delta_h.py` | **Shared** (extracted from GDN) |
| `chunk_fwd_o.py` | `fla/ops/gla/chunk.py` | **Shared** (extracted from GLA, renamed) |

---

## 3. Public API

### `chunk_kda(...)`

Chunk-parallel KDA attention (training mode). The primary entry point for batch training.

```python
from fla.ops.triton.triton_core.kda import chunk_kda

o, final_state = chunk_kda(
    q, k, v, g, beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    safe_gate=False,
    lower_bound=None,
    disable_recompute=False,
    return_intermediate_states=False,
    cp_context=None,          # Not supported — raises NotImplementedError
    transpose_state_layout=False,
    # When use_gate_in_kernel=True, pass via kwargs:
    A_log=...,               # shape [H], log-space decay parameter
    dt_bias=...,             # shape [H*K], optional bias
)
```

### `fused_recurrent_kda(...)`

Fused recurrent KDA for autoregressive decode (inference mode).

```python
from fla.ops.triton.triton_core.kda import fused_recurrent_kda

o, final_state = fused_recurrent_kda(
    q, k, v, g, beta,
    A_log=..., dt_bias=...,  # gate parameters
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=True,
    lower_bound=None,
    cu_seqlens=None,
    transpose_state_layout=False,
)
```

### `fused_kda_gate(...)`

Fused gate computation with autograd support.

```python
from fla.ops.triton.triton_core.kda.gate import fused_kda_gate

g_decayed = fused_kda_gate(g, A_log, dt_bias=dt_bias, lower_bound=None)
```

### Reference Implementations

```python
from fla.ops.triton.triton_core.kda.naive import naive_recurrent_kda, naive_chunk_kda
from fla.ops.triton.triton_core.kda.gate import naive_kda_gate, naive_kda_lowerbound_gate
```

Used as ground-truth for correctness testing.

---

## 4. KDA vs GDN — Key Differences

| Aspect | GDN (Gated Delta Rule) | KDA (Kimi Delta Attention) |
|--------|----------------------|---------------------------|
| **Gate computation** | Simple log-sigmoid or linear decay | `−exp(A_log) · softplus(g + dt_bias)` — learnable per-head decay rate (A_log) with per-head-dim bias (dt_bias) |
| **Intra-chunk attention** | Standard `A = Q·Kᵀ` | Dual matrix: `Aqk = Q·Kᵀ` + `Akk = β·Kᵀ·K` |
| **Backward dAv** | Standard `dv = Aᵀ·do` | KDA-specific `dAv` kernel |
| **Backward dqkg** | Separate dq, dk, dg kernels | Fused `dqkg_fused` kernel |
| **WY recomputation** | Standard GDN WY | KDA WY includes `g_kk` decay term |
| **Decode kernel** | GDN fused recurrent | KDA fused recurrent with dt-bias gate logic |
| **`safe_gate` mode** | Not applicable | M=16 TensorCore acceleration when gate ∈ [−5, 0) |
| **`lower_bound` gate** | Not applicable | Alternative gate: `LB · σ(exp(A_log) · g)`, bounds output |

---

## 5. Architecture Decisions

### Why 100% Triton (no AscendC)?

1. **Interface safety**: Direct migration preserves the original function signatures. AscendC ops have different calling conventions that would require per-kernel parameter mapping.
2. **KDA WY difference**: The KDA `wy_fast.py` includes a `g_kk` decay term absent from the GDN AscendC `npu_recompute_w_u_fwd`. Mixing would cause silent correctness bugs.
3. **Maintenance**: Staying in sync with upstream FLA is straightforward when all kernels are Triton. AscendC replacements would fork the logic.
4. **Incremental optimization path**: Once functional correctness is verified, individual hot-path kernels can be selectively replaced with AscendC versions.

### Context Parallel (CP)

CP is **not supported** in this NPU version. `cp_context` is retained as a placeholder parameter in `chunk_kda()` and `chunk_kda_fwd/bwd()` but will raise `NotImplementedError` if used. CP adaptation is planned as a follow-up task.

---

## 6. NPU Compatibility Notes

### Autotune Configuration

Original CUDA autotune configs (e.g., `num_warps=[2,4,8,16]`, `num_stages=[2,3,4]`, multiple `BK/BV/BT` sizes) are **commented out** and replaced with single conservative configs suitable for NPU triton-ascend. This ensures correctness first; performance tuning is a separate effort.

### CoreDim Limit

NPU (Ascend 910B) has a coreDim limit of 65534 per kernel launch grid. The `filter_safe_configs()` utility (from `..utils`) is used in `gate.py` and `chunk_intra.py` to filter out autotune configs that would exceed this limit at runtime.

### Softplus Implementation

The `softplus` function in `..utils` uses the pure Triton implementation (`tl.math.log(1 + tl.math.exp(x))`), not the NVIDIA PTX inline assembly version. This is verified to work on NPU triton-ascend.

### `tl.gather` Support

`IS_GATHER_SUPPORTED` is auto-detected at import time. `chunk_intra.py` uses a fallback path when `tl.gather` is not available on the target backend.

### TF32

`IS_TF32_SUPPORTED` is set to `False` on NPU. The `SOLVE_TRIL_DOT_PRECISION` in `chunk_intra.py` falls back to `ieee` mode.

---

## 7. Testing

Tests are located at `tests/triton/test_kda_chunk.py` and use `naive_recurrent_kda` as the ground-truth reference.

**Test cases** (all Megatron deployment parameters):

| Test | What it verifies |
|------|-----------------|
| `test_megatron_forward` | Forward output precision vs naive (H∈{32,64}, safe_gate, lower_bound, dtype) |
| `test_megatron_backward` | Forward+backward gradient precision (including T=8192) |
| `test_megatron_varlen` | Variable-length sequence forward+backward |
| `test_megatron_param_init_ranges` | A_log and dt_bias initialization range checks |
| `test_megatron_realistic_shapes` | Smoke test with production shapes |
| `test_megatron_determinism` | Output determinism across calls |

**Run:**
```bash
pytest tests/triton/test_kda_chunk.py -v
```

---

## 8. Dependency Map

```
chunk_kda (chunk.py)
├── chunk_fwd.py
│   ├── chunk_delta_h.py       → chunk_gated_delta_rule_fwd_h
│   ├── chunk_fwd_o.py         → chunk_kda_fwd_o_gk
│   ├── chunk_intra.py         → chunk_kda_fwd_intra
│   ├── gate.py                → kda_gate_chunk_cumsum
│   └── ..cumsum.py            → chunk_local_cumsum
├── chunk_bwd.py
│   ├── chunk_delta_h.py       → chunk_gated_delta_rule_fwd_h, chunk_gated_delta_rule_bwd_dhu
│   ├── chunk_intra.py         → chunk_kda_bwd_intra
│   ├── gate.py                → kda_gate_bwd, kda_gate_chunk_cumsum
│   ├── wy_fast.py             → recompute_w_u_fwd
│   └── ..cumsum.py            → chunk_local_cumsum
├── ..l2norm.py                → l2norm_fwd, l2norm_bwd
└── ..utils.py                 → input_guard, autocast_custom_fwd/bwd, prepare_chunk_indices, ...

fused_recurrent_kda (fused_recurrent.py)
├── ..utils.py                 → exp, softplus, input_guard
└── (self-contained Triton kernel)
```

External dependencies (from `..utils` and `..cumsum`):

| Dependency | Provides |
|-----------|----------|
| `utils.input_guard` | Tensor contiguity/layout guards |
| `utils.autocast_custom_fwd/bwd` | Autocast context managers |
| `utils.prepare_chunk_indices` | Varlen chunk index computation |
| `utils.check_shared_mem` | GPU shared memory detection |
| `utils.autotune_cache_kwargs` | Triton autotune caching |
| `utils.filter_safe_configs` | NPU coreDim-safe autotune filtering |
| `utils._get_autotune_layer` | Autotune layer extraction for filter_safe_configs |
| `utils.exp`, `utils.exp2` | Triton exp/exp2 with NPU compatibility |
| `utils.softplus` | NPU-compatible softplus (no PTX) |
| `utils.IS_GATHER_SUPPORTED` | Runtime `tl.gather` availability |
| `utils.IS_TF32_SUPPORTED` | Always `False` on NPU |
| `utils.gather` | `tl.gather` with fallback |
| `utils.RCP_LN2` | `1 / ln(2)` constant |
| `cumsum.chunk_local_cumsum` | Chunk-local cumulative sum |

---

## 9. Known Limitations

1. **Performance**: Autotune configs are conservative (single config per kernel). Performance tuning for Ascend 910B is not yet done.
2. **No CP**: Context Parallel is not supported. `cp_context` parameter exists but raises `NotImplementedError`.
3. **No CUTLASS/FlashKDA backend**: Only Triton chunk and fused_recurrent modes are available.
4. **`return_intermediate_states`**: Supported but requires `torch.inference_mode()` and `disable_recompute=False`.
5. **Head dim limit**: `K ≤ 256` (asserted in `chunk_kda`).

---

## 10. Migration Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Utility function supplements (`softplus`, `filter_safe_configs`, etc.) | ✅ |
| 2 | KDA-specific Triton kernel migration | ✅ |
| 3 | Shared Triton operator extraction (`chunk_delta_h`, `chunk_fwd_o`) | ✅ |
| 4 | Orchestration layer adaptation (`chunk.py`, `chunk_fwd.py`, `chunk_bwd.py`) | ✅ |
| 5 | End-to-end integration script | ⏭️ Skipped (pytest covers it) |
| 6 | Test migration (`test_kda_chunk.py`) | ✅ |
| 7 | Documentation & cleanup | ✅ |
