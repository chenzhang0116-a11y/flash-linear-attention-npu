# KDA (Kimi Delta Attention) — Triton-only NPU implementation
from .kda import (
    chunk_kda,
    chunk_gated_delta_rule_fwd_h,
    chunk_gated_delta_rule_bwd_dhu,
    chunk_kda_fwd_o_gk,
    fused_recurrent_kda,
)

__all__ = [
    "chunk_kda",
    "chunk_gated_delta_rule_fwd_h",
    "chunk_gated_delta_rule_bwd_dhu",
    "chunk_kda_fwd_o_gk",
    "fused_recurrent_kda",
]
