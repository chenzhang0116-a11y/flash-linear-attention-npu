# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Adapted from fla/ops/kda/chunk_fwd.py for NPU (Triton-only, no CP support).

import torch

from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
from .chunk_fwd_o import chunk_kda_fwd_o_gk
from .chunk_intra import chunk_kda_fwd_intra
from .gate import kda_gate_chunk_cumsum
from ..cumsum import chunk_local_cumsum
from ..utils import RCP_LN2


def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    cp_context=None,  # CP not supported in NPU version; kept as placeholder
    transpose_state_layout: bool = False,
):
    # Apply gate activation
    g_org = None
    if use_gate_in_kernel:
        g_org = g
        g = kda_gate_chunk_cumsum(
            g=g_org,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=RCP_LN2,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            lower_bound=lower_bound,
        )
    else:
        g = chunk_local_cumsum(
            g=g,
            scale=RCP_LN2,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices
        )

    # qg = None if disable_recompute is False
    w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
        disable_recompute=disable_recompute
    )

    if cp_context is not None:
        raise NotImplementedError(
            "Context Parallel (CP) is not supported in the NPU Triton-only version of KDA. "
            "Please set cp_context=None."
        )

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_indices=chunk_indices,
        use_exp2=True,
        transpose_state_layout=transpose_state_layout,
    )

    o = chunk_kda_fwd_o_gk(
        q=q,
        v=v_new,
        g=g,
        A=Aqk,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        use_exp2=True,
        transpose_state_layout=transpose_state_layout,
    )
    if disable_recompute is False:
        # Delete to save memory
        w, u, qg, kg, v_new = None, None, None, None, None
        if not return_intermediate_states:
            # Only delete h if not requested for inference
            h = None
        if use_gate_in_kernel:
            g = None
    return o, final_state, g, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state