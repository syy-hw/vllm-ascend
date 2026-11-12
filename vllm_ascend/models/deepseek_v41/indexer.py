# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 lightning indexer.

Structurally different from the V4 (V4-Flash) indexer -- do not reuse
``vllm_ascend.models.deepseek_v4.indexer`` for V4.1. Official semantics
(reference model.py, Indexer):

- Weights: ``wq_b`` (q_lora -> n_heads * index_head_dim) and ``weights_proj``
  (hidden -> n_heads) on every index source layer; the four ``kv_source``
  layers additionally own ``wk`` (head_dim -> index_head_dim) plus ``k_norm``
  and therefore the shared index K cache ("owns_k").
- owns_k layers derive index keys from their own compressor latent (pre-RoPE
  output): k = k_norm(wk(latent)); the last ``rope_head_dim`` lanes rotate at
  the group position j * ratio.
- Scoring: q = wq_b(qr) (last rope lanes rotated) einsum with K -> relu ->
  multiplied by weights_proj(x) * softmax_scale * n_heads**-0.5 -> summed
  over heads.
- Two-level top-k: the candidate source layer (20) keeps the best
  ``candidate_topk_blocks`` blocks (block = ``candidate_block_size``
  compressed positions, scored by block max, newest block pinned); layers
  above it mask everything outside those blocks before their own top-k.
- topk = min(index_topk, visible compressed positions), re-sorted into
  position order; unreachable positions -> -1, the rest shifted by the
  window-KV offset.

This file provides the module structure plus a correctness-first pure-torch
forward. The fused NPU operator path
(``npu_vllm_quant_lightning_indexer``) is wired in a later pass; when it is,
``cmp_ratio`` must be the layer's actual ratio (1 or 2, not the V4 4).
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

# e2m1 magnitudes and the ue8m0 group size of the index cache
_FP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP4_GRID_CACHE: dict = {}  # device -> (boundaries, lower grid, upper grid)


def _ceil_log2(x: torch.Tensor) -> torch.Tensor:
    """ceil(log2(x)) for positive floats, exact at powers of two.

    Bit-level IEEE754 decomposition. torch.frexp has no NPU kernel (it
    silently falls back to CPU — a per-call device->host->device round trip
    on a hot path, and a graph-capture blocker), so read the exponent field
    and the mantissa-bits-nonzero flag straight from the bit pattern:
    x = 1.f * 2**e  =>  ceil(log2 x) = e if f == 0 else e + 1.

    WARNING (engine graph-replay finding, 2026-11-10): this view+int-op
    chain is NOT replay-safe inside the engine's compiled decode — the
    acl concrete graph loses the dependency edges around the int32 bitcast
    and the int kernels read UNINITIALIZED memory at replay (graph_qp4
    dissect: amax/div fresh and correct, e garbage like 126/-53/93).
    Graph-executed call sites must use _ceil_log2_float instead.
    """
    bits = x.contiguous().view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127
    frac_nonzero = (bits & 0x7FFFFF) != 0
    return torch.where(frac_nonzero, (e + 1).to(x.dtype), e.to(x.dtype))


def _ceil_log2_float(x: torch.Tensor) -> torch.Tensor:
    """ceil(log2(x)) via pure float ops — replay-safe in the compiled decode.

    Canonical form e = where(x <= 2**e0, e0, e0+1) with e0 =
    floor(fl(log2 x)); exp2 of an integer is exact, so the comparison
    self-corrects both log2 rounding directions at integer boundaries.
    The residual deviation vs the bit-exact form (e0 one too low when
    fl(log2) rounds down across an integer for x just above a power of
    two) lands on the SMALL-scale side, where the callers' +-448 clamp
    already bounds the values — and eager and the replayed graph run the
    SAME fp32 log2 kernel, so both modes stay bit-identical (the kvprobe
    bit-exact gate checks exactly that).
    """
    e0 = torch.floor(torch.log2(x))
    return torch.where(x <= torch.exp2(e0), e0, e0 + 1.0)


def fp4_fake_quant_inplace(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Per-32-block FP4-E2M1 quantize -> dequantize, ue8m0 (power-of-2) scales.

    Mirrors the official ``fp4_act_quant(x, 32, inplace=True)`` used on both
    indexer queries and keys: amax per block, scale = 2 ** ceil(log2(amax/6))
    (never zero: clamped at 6 * 2**-126), values snapped to the e2m1 grid and
    scaled back. Tie-breaking at grid midpoints follows round-half-to-even;
    device kernels are the final numeric authority.
    """
    orig_shape = x.shape
    xq = x.reshape(-1, block_size).float()
    amax = xq.abs().amax(dim=-1, keepdim=True).clamp_min(6.0 * 2.0**-126)
    scale = torch.pow(2.0, _ceil_log2_float(amax / 6.0))
    normalized = (xq / scale).clamp(-6.0, 6.0)
    magnitude = normalized.abs()
    # graph-capture safety (same as fp4_fake_quant_inplace_e4m3scale): the
    # grid constants must NOT be built from Python lists inside a captured
    # region (tensor-from-list is an H2D copy). Cache per device once.
    cache = _FP4_GRID_CACHE.get(x.device)
    if cache is None:
        cache = (
            torch.tensor(_FP4_MAGNITUDES[1:-1], device=x.device, dtype=torch.float32),
            torch.tensor(_FP4_MAGNITUDES[:-1], device=x.device, dtype=torch.float32),
            torch.tensor(_FP4_MAGNITUDES[1:], device=x.device, dtype=torch.float32),
        )
        _FP4_GRID_CACHE[x.device] = cache
    boundaries, grid_lower, grid_upper = cache
    # boundaries between consecutive e2m1 magnitudes; pick nearest, ties down
    idx = torch.bucketize(magnitude, boundaries, right=False).to(torch.float32)
    # exact midpoints round to the even neighbor (sign bit untouched)
    lower = grid_lower[idx.long()]
    upper = grid_upper[idx.long()]
    midpoint = (lower + upper) / 2
    choose_upper = magnitude > midpoint
    # torch.sign lowers to aten::copysign on this backend, which has no NPU
    # kernel (CPU fallback on the hot path); a plain select avoids it. The
    # magnitude is 0 exactly where normalized == 0, so the +/-1 sign never
    # changes the quantized value there.
    sgn = torch.where(normalized >= 0, 1.0, -1.0)
    quantized = torch.where(choose_upper, upper, lower) * sgn
    out = (quantized * scale).reshape(orig_shape)
    x.copy_(out.to(x.dtype))
    return x


def fp4_fake_quant_e4m3scale(x: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """Out-of-place twin of fp4_fake_quant_inplace_e4m3scale (same math).

    The inplace variant's trailing ``x.copy_`` is DROPPED by the npugraph_ex
    FX functionalization inside the compiled decode graph (rows reached the
    cmp pool UNQUANTIZED); the eager post-replay drain uses this pure form
    to quantize the exported row before staging it back.
    """
    import functools

    # run the SAME implementation on a throwaway copy, then return the value
    # (single source of truth: no duplicated math)
    y = fp4_fake_quant_inplace_e4m3scale(x.clone())
    return y


def fp8_fake_quant_inplace_ue8m0(x: torch.Tensor, block_size: int = 32,
                                  probe: dict | None = None) -> torch.Tensor:
    """Per-32-block FP8-E4M3 quantize -> dequantize with UE8M0 (power-of-2) scales.

    Mirrors the official ``act_quant(x, 32, "ue8m0", e8m0, inplace=True)``
    round trip applied to the window KV (post-RoPE) and to the o activation
    (post inverse-RoPE) in the official Attention.forward: per-32-block amax
    (floored at 1e-4), scale = 2 ** ceil(log2(amax / 448)) stored as ue8m0,
    values snapped to the e4m3 grid via the dtype cast (round-to-nearest-even)
    and scaled back to the input dtype.

    ``probe`` (first-divergence experiment): persistent buffers recording the
    per-stage intermediates of token 0 (qamax/qscale/qq/qout) via plain device
    copies, so the replayed graph re-records them each step; the layer-IO
    capture copies them into the per-layer entry and the drain writes files.
    """
    orig_shape = x.shape
    xq = x.reshape(-1, block_size).float()
    amax = xq.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    # bit-level ceil(log2): torch.frexp falls back to CPU on NPU (hot path,
    # once per layer per decode step since patch 80)
    amaxdiv = amax / 448.0
    e = _ceil_log2_float(amaxdiv)
    # The exponent MUST be cast to float32 before the pow: _ceil_log2 returns
    # an INT32 tensor, and in the engine's compiled decode the lowered
    # integer-power kernel computes 2**negative == 0. The cast keeps the
    # official act_quant value semantics (power-of-two scale) while forcing
    # the float kernel (eager bit-exact before and after; fix_equiv_gtest).
    scale = torch.pow(2.0, e.to(torch.float32))
    q = (xq / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    out = (q * scale).reshape(orig_shape)
    if probe is not None:
        # independent per-stage buffers; later in-place steps (x.copy_)
        # touch none of them, so earlier records are never overwritten.
        _n = int(probe["qamax"].shape[0])
        probe["qamax"].copy_(amax[:_n])
        probe["qdiv"].copy_(amaxdiv[:_n])
        probe["qexp"].copy_(e[:_n].to(torch.float32))
        probe["qscale"].copy_(scale[:_n])
        probe["qq"].copy_(q[:_n])
        probe["qout"].copy_(
            out.reshape(-1, block_size)[:_n])
    x.copy_(out.to(x.dtype))
    return x


def fp8_fake_quant_ue8m0(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Out-of-place variant of fp8_fake_quant_inplace_ue8m0.

    The inplace variant ends with ``x.copy_(out)`` — under the npugraph_ex
    FX functionalization that mutation on the flowing kv tensor was DROPPED
    in the engine's compiled decode (cache rows got UNQUANTIZED values while
    eager rows were fp8-rounded), and cloning the input first collapsed the
    value to ~0 instead. This pure variant returns the rounded tensor with
    no in-place op anywhere, which is functionalization-safe by
    construction.
    """
    orig_shape = x.shape
    xq = x.reshape(-1, block_size).float()
    amax = xq.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.pow(2.0, _ceil_log2_float(amax / 448.0))
    q = (xq / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    return (q * scale).reshape(orig_shape).to(x.dtype)


def fp4_fake_quant_inplace_e4m3scale(x: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """Per-16-block FP4-E2M1 quantize -> dequantize with E4M3 scales.

    Mirrors the official compressed-KV fake quant
    ``fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)``:
    scale = amax / 6 cast to e4m3 (amax floor 6 * 2**-9 keeps it nonzero and
    exactly representable), values snapped to the e2m1 grid, scaled back.
    """
    orig_shape = x.shape
    xq = x.reshape(-1, block_size).float()
    amax = xq.abs().amax(dim=-1, keepdim=True).clamp_min(6.0 * 2.0**-9)
    scale = (amax / 6.0).to(torch.float8_e4m3fn).float()
    scale = torch.where(scale > 0, scale, torch.full_like(scale, 2.0**-9))
    normalized = (xq / scale).clamp(-6.0, 6.0)
    magnitude = normalized.abs()
    # graph-capture safety: the grid constants must NOT be created from
    # Python lists inside a captured region (tensor-from-list is an H2D
    # copy). Cache per-device constant tensors once.
    cache = _FP4_GRID_CACHE.get(x.device)
    if cache is None:
        cache = (
            torch.tensor(_FP4_MAGNITUDES[1:-1], device=x.device, dtype=torch.float32),
            torch.tensor(_FP4_MAGNITUDES[:-1], device=x.device, dtype=torch.float32),
            torch.tensor(_FP4_MAGNITUDES[1:], device=x.device, dtype=torch.float32),
        )
        _FP4_GRID_CACHE[x.device] = cache
    boundaries, grid_lower, grid_upper = cache
    idx = torch.bucketize(magnitude, boundaries, right=False).to(torch.float32)
    lower = grid_lower[idx.long()]
    upper = grid_upper[idx.long()]
    midpoint = (lower + upper) / 2
    choose_upper = magnitude > midpoint
    # torch.sign lowers to aten::copysign on this backend (no NPU kernel,
    # CPU fallback); a plain select is identical: the magnitude is 0 exactly
    # where normalized == 0, so the +/-1 sign cannot change the value there.
    sgn = torch.where(normalized >= 0, 1.0, -1.0)
    quantized = torch.where(choose_upper, upper, lower) * sgn
    out = (quantized * scale).reshape(orig_shape)
    x.copy_(out.to(x.dtype))
    return x


def fp4_fake_quant_e4m3scale(x: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """Out-of-place twin of fp4_fake_quant_inplace_e4m3scale (same math).

    The inplace variant's trailing ``x.copy_`` is DROPPED by the npugraph_ex
    FX functionalization inside the compiled decode graph (rows reached the
    cmp pool UNQUANTIZED); the eager post-replay drain uses this pure form
    to quantize the exported row before staging it back.
    """
    import functools

    # run the SAME implementation on a throwaway copy, then return the value
    # (single source of truth: no duplicated math)
    y = fp4_fake_quant_inplace_e4m3scale(x.clone())
    return y


def cmp_kv_from_latent(indexer, latents: torch.Tensor, token_pos: torch.Tensor) -> torch.Tensor:
    """Compressed-KV row from a pooled AND RMSNorm-ed latent (official
    Compressor.forward -> write_cache math): RoPE the tail rope lanes at the
    group-start token position, cast bf16, then per-16 fp4 fake-quant with
    E4M3 scales (official fp4_act_quant(..., 16, True, scale_dtype=e4m3)).

    Single implementation shared by the eager reference write and the graph
    decode scatter so the two paths cannot drift apart again (the raw-pooled
    variant bug: pool output must be normed BEFORE this step — the reference
    pool and official Compressor.forward both return RMSNorm-ed latents).
    """
    rd = indexer.rope_head_dim
    base_lanes = latents[:, : latents.shape[-1] - rd]
    rot_lanes = indexer._rotary_at_positions(
        latents[:, latents.shape[-1] - rd :].contiguous(), token_pos)
    cmp_kv = torch.cat([base_lanes.float(), rot_lanes.float()], dim=-1).to(torch.bfloat16)
    fp4_fake_quant_inplace_e4m3scale(cmp_kv, block_size=16)
    return cmp_kv


def graph_safe_candidate_blocks(score: torch.Tensor, akl: torch.Tensor,
                                block_size: int, topk_blocks: int) -> torch.Tensor:
    """Candidate-block mask (official select_candidate_blocks semantics) with
    no sort-family op: per-block amax, pin the block holding each query's
    newest VISIBLE position (akl-1)//block_size, keep the top ``topk_blocks``
    blocks by graph_safe_topk's selection, expand to positions via gather.

    ``score`` [B, W] with invisible positions at -inf; ``akl`` [B] int-ish
    visible-length per request. Returns a bool [B, W] position mask (True =
    inside a kept candidate block); blocks that are entirely unreachable
    (amax -inf) never survive, matching the official ``values > -inf`` gate.
    """
    B, W = score.shape
    nblk = W // block_size
    blk = score.view(B, nblk, block_size).amax(dim=-1)
    last = (akl.reshape(B, 1).to(torch.int64) - 1) // block_size
    blk = blk.masked_fill(
        torch.arange(nblk, device=score.device).view(1, -1) == last, torch.inf)
    _, sel = graph_safe_topk(blk, min(topk_blocks, nblk))
    # official gate: keep = top.values > -inf — the PINNED block scores +inf
    # (kept: inf > -inf) and fully-unreachable blocks score -inf (dropped).
    # isfinite would wrongly drop the pinned +inf block.
    keep = sel & (blk > -float("inf"))
    pos_blk = torch.arange(W, device=score.device) // block_size
    return keep.gather(-1, pos_blk.view(1, -1).expand(B, -1))


def graph_safe_topk(score: torch.Tensor, k: int):
    """Exact descending top-k WITHOUT any sort-family op (torch.topk and
    argsort corrupt ACL full-graph replay on this stack — bisected in
    cycles 35-47). Only elementwise / compare / sum / cumsum / matmul ops.

    Returns (idxs [B, k] int64 in POSITION-ASCENDING order — the official
    select_topk output order, valid positions with non-finite score mapped
    to -1 by the caller via isfinite, matching the reference validity gate).

    Semantics vs torch.topk:
      - the selected SET is exactly the k largest by value;
      - ties at the k-boundary break by POSITION ORDER (torch's internal
        tie order is unspecified and platform-dependent; finite fp32 ties
        are measure-zero for real scores, and -inf "ties" are all invalid
        so their identity never reaches the output).

    Algorithm:
      1. order-preserving int64 key of the float bits; 33-iteration integer
         bisection of the exact k-th largest key (compare + sum only);
      2. gt (> t) all selected; among the == t ties keep the first (k - m)
         by position (inclusive cumsum rank);
      3. extraction: the selected-position rank cumsum is non-decreasing,
         so the j-th selected position is recovered exactly by a [k, W]
         one-hot matmul against arange (fp32 holds ints < 2^24 exactly).
    """
    B, W = score.shape
    bits = score.contiguous().view(torch.int32)
    # order-preserving signed-int64 key of the float bit pattern:
    # negative floats (sign bit set): ~bits clears the sign bit and REVERSES
    # the order into ascending (more negative float -> larger bits -> smaller
    # ~bits); non-negative floats: bits + 2**31 lands strictly above the
    # whole negative branch. XOR with the sign bit would keep UNSIGNED order
    # but reinterprets as negative int32 — wrong for signed compares.
    key = torch.where(
        bits < 0,
        (~bits).to(torch.int64),
        bits.to(torch.int64) + 2**31,
    )
    # key space is [0, 2**32) (unsigned-order semantics in signed int64)
    lo = torch.zeros(B, dtype=torch.int64, device=score.device)
    hi = torch.full((B,), 2**32, dtype=torch.int64, device=score.device)
    for _ in range(33):
        mid = (lo + hi + 1) >> 1
        cnt = (key >= mid.unsqueeze(-1)).sum(-1, dtype=torch.int64)
        up = cnt >= k
        lo = torch.where(up, mid, lo)
        hi = torch.where(up, hi, mid - 1)
    t = lo
    ge = key >= t.unsqueeze(-1)
    gt = key > t.unsqueeze(-1)
    m = gt.sum(-1, dtype=torch.int64)
    tie = ge & ~gt
    tie_rank = tie.to(torch.int32).cumsum(-1)
    keep_tie = tie & (tie_rank <= (k - m).clamp(min=0).view(-1, 1))
    sel = gt | keep_tie
    sel_rank = sel.to(torch.int32).cumsum(-1)
    arange_k = torch.arange(1, k + 1, device=score.device, dtype=torch.float32)
    # NOTE the & sel: unselected positions CARRY the stale rank value (the
    # cumsum only advances at selections), so the == alone matches long runs
    # of unselected positions and the matmul sums garbage positions.
    onehot = (
        sel.unsqueeze(1)
        & (sel_rank.unsqueeze(1) == arange_k.view(1, k, 1))
    ).to(torch.float32)
    posf = torch.arange(W, device=score.device, dtype=torch.float32).unsqueeze(0)
    idxs = torch.matmul(onehot, posf.unsqueeze(-1)).squeeze(-1)
    return idxs.to(torch.int64), sel


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the two-level top-k: keep the ``topk_blocks`` highest-scoring blocks per query.

    ``logits`` is [..., n_positions] with positions the query cannot reach
    already at -inf, which is what makes a block score of -inf mean "not
    reachable yet". ``compress_lens`` is a plain int during decode, or
    broadcasts against logits' leading dims during prefill. Returns a bool
    mask shaped like ``logits``, so the layers consuming it just mask and
    never think about blocks again.
    """
    width = logits.size(-1)
    # score each block by its best position; -inf pads the last one out to block_size
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    # the block with this query's newest position is only partly filled, so
    # pin it in: it holds the most recent tokens but could otherwise be
    # outscored by an older, full block
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)

    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    # fewer reachable blocks than topk_blocks means leftover picks came back
    # -inf: drop them
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def _reference_linear_weight(weight: torch.Tensor) -> torch.Tensor:
    """Guard the torch reference path against quantized compressor weights.

    The fused compressor op consumes the raw (possibly block-quantized)
    weight; the reference matmul cannot reconstruct it from the tensor alone.
    A5 (which the V4.1 path targets) builds the compressor linears with
    ``quant_config=None``, so this only fires on unsupported configurations.
    """
    if weight.dtype in (torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2):
        raise NotImplementedError(
            "V4.1 reference compressor pooling needs unquantized (bf16/fp32) compressor weights"
        )
    return weight


def _reference_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Official RMSNorm: fp32 reduction, weight multiply, cast back."""
    dtype = x.dtype
    xf = x.float()
    var = xf.square().mean(dim=-1, keepdim=True)
    xn = xf * torch.rsqrt(var + eps)
    return (weight.float() * xn).to(dtype)


class ReferenceCompressorPool:
    """Persistent reference implementation of the official Compressor pooling.

    Implements official-ref ``model.py`` ``Compressor.forward`` for one
    owns_k layer: ``kv = x @ wkv.T`` and ``score = x @ wgate.T`` in fp32,
    softmax over each complete group of ``ratio`` tokens, weighted sum, then
    RMSNorm (``norm_weight``, ``eps``) on the emitted latents — norm comes
    AFTER pooling, matching the official ``self.norm(kv.to(dtype))``.

    The partial group at a chunk tail is carried in per-request fp32 state
    buffers (official ``kv_state``/``score_state``). Request identity is the
    first KV block id (``block_table[i, 0]``), stable for a request's
    lifetime; slot positions are derived from ``start_pos % ratio`` exactly
    like the official carry, so decode (1 token/req), prefill-from-0 (split
    complete groups, save remainder) and chunked-prefill continuation (a
    group straddling the chunk boundary completes from the state) all follow
    one code path. ``ratio == 1`` is a plain ``norm(wkv(x))`` per token with
    no state and no gate (official ratio-1 compressor; the checkpoint has no
    wgate for the ratio-1 source layer).
    """

    def __init__(self, compress_ratio: int, head_dim: int, eps: float, max_entries: int = 4096):
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.eps = eps
        self.max_entries = max_entries
        # per-request pooling state (official kv_state / score_state)
        self.kv_state: torch.Tensor | None = None  # fp32 [rows, ratio, head_dim]
        self.score_state: torch.Tensor | None = None  # fp32 [rows, ratio, head_dim]
        self.row_of_req: dict[int, int] = {}
        # contiguous-history length per request key: the carry (kv/score
        # state) is only valid when the previous chunk ended exactly where
        # this one starts (decode continuation or chunked prefill).
        self.row_len: dict[int, int] = {}
        self.row_free: list[int] = []
        self.next_row = 0
        self.row_lru: OrderedDict[int, None] = OrderedDict()

    # ------------------------------------------------------------------
    # state-buffer management
    # ------------------------------------------------------------------
    def _grow_state_rows(self, device: torch.device, min_rows: int) -> None:
        new_cap = max(min_rows, 1)
        if self.kv_state is not None:
            new_cap = max(new_cap, self.kv_state.shape[0] * 2)
        kv = torch.zeros(new_cap, self.compress_ratio, self.head_dim, dtype=torch.float32, device=device)
        score = torch.full(
            (new_cap, self.compress_ratio, self.head_dim), -torch.inf, dtype=torch.float32, device=device
        )
        if self.kv_state is not None:
            n = self.kv_state.shape[0]
            kv[:n].copy_(self.kv_state)
            score[:n].copy_(self.score_state)
        self.kv_state, self.score_state = kv, score

    def _reset_row(self, row: int) -> None:
        self.kv_state[row].zero_()
        self.score_state[row].fill_(-torch.inf)

    def _row_for(self, req_key: int, start_pos: int, device: torch.device) -> int:
        row = self.row_of_req.get(req_key)
        if row is not None:
            self.row_lru.move_to_end(req_key)
            prev_len = self.row_len.get(req_key, start_pos)
            if start_pos == 0:
                # a fresh request reusing a finished request's block id:
                # reset the carry so stale slots cannot leak into group 0
                self._reset_row(row)
            elif prev_len != start_pos:
                # block id reused by a request whose schedule does not start
                # where the previous one ended (eviction / prefix-cache mix):
                # the partial-group carry belongs to foreign tokens; drop it.
                self._reset_row(row)
            return row
        if self.row_free:
            row = self.row_free.pop()
        else:
            if self.kv_state is None or self.next_row >= self.kv_state.shape[0]:
                self._grow_state_rows(device, self.next_row + 1)
            row = self.next_row
            self.next_row += 1
        self._reset_row(row)
        self.row_of_req[req_key] = row
        self.row_lru[req_key] = None
        while len(self.row_lru) > self.max_entries:
            victim, _ = next(iter(self.row_lru.items()))
            self.row_lru.pop(victim)
            self.row_free.append(self.row_of_req.pop(victim))
            self.row_len.pop(victim, None)
        return row

    # ------------------------------------------------------------------
    # pooling
    # ------------------------------------------------------------------
    def pool(
        self,
        hidden_states: torch.Tensor,
        wkv_weight: torch.Tensor,
        wgate_weight: torch.Tensor | None,
        norm_weight: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        start_pos: torch.Tensor,
        req_keys: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool this step's tokens into completed-group latents.

        Returns ``(latents [n_complete, head_dim], req_idx int64 [n_complete],
        group_pos int64 [n_complete])`` where ``group_pos`` is the global
        compressed position ``(start_pos + offset_in_req) // ratio`` of each
        completed group. ``req_keys`` are the per-request first block ids the
        persistent state is keyed by (the dsa bridge derives them from
        ``block_table[:, 0]``).
        """
        T = hidden_states.shape[0]
        dtype = hidden_states.dtype
        device = hidden_states.device
        ratio = self.compress_ratio
        B = query_start_loc.shape[0] - 1

        qsl_cpu = query_start_loc.detach().cpu().to(torch.int64).tolist()
        sp_cpu = start_pos.detach().cpu().to(torch.int64).tolist() if start_pos is not None else [0] * B

        if ratio == 1:
            # official: one token per group, no gate, no fp32, norm per token
            kv = F.linear(hidden_states, _reference_linear_weight(wkv_weight))
            latents = _reference_rms_norm(kv, norm_weight, self.eps)
            lens = query_start_loc[1:] - query_start_loc[:-1]
            req_idx = torch.repeat_interleave(torch.arange(B, device=device), lens, output_size=T)
            local = torch.arange(T, device=device) - query_start_loc[req_idx].long()
            group_pos = local if start_pos is None else start_pos[req_idx].long() + local
            return latents, req_idx, group_pos

        if wgate_weight is None:
            raise ValueError("ratio>1 reference pooling requires the compressor wgate weight")

        x = hidden_states.float()
        kv_all = F.linear(x, _reference_linear_weight(wkv_weight).float())
        score_all = F.linear(x, _reference_linear_weight(wgate_weight).float())

        lat_parts: list[torch.Tensor] = []
        req_parts: list[torch.Tensor] = []
        gpos_parts: list[torch.Tensor] = []
        for i in range(B):
            t0, t1 = qsl_cpu[i], qsl_cpu[i + 1]
            n = t1 - t0
            if n == 0:
                continue
            s0 = sp_cpu[i]
            req_key = int(req_keys[i]) if req_keys is not None else i
            row = self._row_for(req_key, s0, device)
            rem = s0 % ratio
            m = (s0 + n) // ratio - s0 // ratio  # groups completing in this chunk
            g_first = s0 // ratio
            head = ratio - rem if m > 0 else 0  # in-chunk tokens of the first completed group
            if m > 0:
                # complete the first (possibly chunk-straddling) group in the
                # state row, pooling state + chunk tokens together
                self.kv_state[row, rem:ratio] = kv_all[t0 : t0 + head]
                self.score_state[row, rem:ratio] = score_all[t0 : t0 + head]
                first = (self.kv_state[row] * self.score_state[row].softmax(dim=0)).sum(dim=0, keepdim=True)
                if m > 1:
                    r0 = t0 + head
                    rest_kv = kv_all[r0 : r0 + (m - 1) * ratio].reshape(m - 1, ratio, self.head_dim)
                    rest_sc = score_all[r0 : r0 + (m - 1) * ratio].reshape(m - 1, ratio, self.head_dim)
                    rest = (rest_kv * rest_sc.softmax(dim=1)).sum(dim=1)
                    pooled = torch.cat([first, rest], dim=0)
                else:
                    pooled = first
                lat_parts.append(_reference_rms_norm(pooled.to(dtype), norm_weight, self.eps))
                req_parts.append(torch.full((m,), i, dtype=torch.int64, device=device))
                gpos_parts.append(torch.arange(g_first, g_first + m, dtype=torch.int64, device=device))
            # carry the trailing partial group in the state row (official remainder)
            tail_start = head + (m - 1) * ratio if m > 0 else 0
            if tail_start < n:
                tail_local = torch.arange(tail_start, n, dtype=torch.int64, device=device)
                slots = (s0 + tail_local) % ratio
                self.kv_state[row].index_copy_(0, slots, kv_all[t0 + tail_start : t1])
                self.score_state[row].index_copy_(0, slots, score_all[t0 + tail_start : t1])
            self.row_len[req_key] = s0 + n

        if not lat_parts:
            return (
                hidden_states.new_zeros((0, self.head_dim)),
                torch.empty(0, dtype=torch.int64, device=device),
                torch.empty(0, dtype=torch.int64, device=device),
            )
        return torch.cat(lat_parts), torch.cat(req_parts), torch.cat(gpos_parts)


class DevicePoolState:
    """Device-side pooling state for DECODE, keyed by request first block id.

    Official ``model.py`` keeps fp32 kv/score state shaped
    ``(max_batch, ratio, head_dim)``. Rows here are indexed by the request's
    FIRST BLOCK ID (``block_table[i, 0]``, stable for the request lifetime —
    the same identity the reference store uses), NOT by batch row number, so
    batch reordering/retirement cannot mix requests' state. A fresh request
    (start_pos == 0) rewrites its slots from group 0 before any group
    completes, making stale rows unreachable.

    Layout note (cache_roundtrip.py, ALL PASS with the real ops): the engine
    indexer cache page = 16384B key region (128 rows stride 128) + 512B
    scale region (ONE fp32 per row); the epilog takes FLAT int32 slots
    ``page * rows_per_page + row``.
    """

    def __init__(self, num_rows: int, compress_ratio: int, head_dim: int,
                 device: torch.device):
        self.ratio = compress_ratio
        self.head_dim = head_dim
        if compress_ratio > 1:
            self.kv = torch.zeros(num_rows, compress_ratio, head_dim,
                                  dtype=torch.float32, device=device)
            self.score = torch.full((num_rows, compress_ratio, head_dim),
                                    -float("inf"), dtype=torch.float32, device=device)
        else:
            self.kv = None
            self.score = None

    def pool_decode(self, x, wkv_weight, wgate_weight, norm_weight, eps,
                    start_pos, slots):
        """One decode token per request, official Compressor.forward math.

        ``x`` [B, hidden] bf16; ``start_pos`` [B] device; ``slots`` [B]
        long device = per-request state rows (engine request slot when the
        persist channel carries it, else first block id).
        Returns (latent [B, head_dim] bf16, complete [B] bool, group [B]).

        The returned latent is RMSNorm-ed AFTER pooling — identical stage to
        ``ReferenceCompressorPool.pool`` and the official
        ``self.norm(kv.to(dtype))``: fp32 linears, fp32 softmax-weighted sum,
        cast to the input dtype, then RMSNorm with ``norm_weight``/``eps``.
        Index keys AND compressed-KV rows both consume THIS tensor.
        """
        B = x.shape[0]
        dev = x.device
        if self.ratio == 1:
            kv = torch.nn.functional.linear(x, wkv_weight)
            latent = _reference_rms_norm(kv, norm_weight, eps)
            return (latent, torch.ones(B, dtype=torch.bool, device=dev),
                    start_pos.long())
        assert wgate_weight is not None
        kv_all = torch.nn.functional.linear(x.float(), wkv_weight.float())
        sc_all = torch.nn.functional.linear(x.float(), wgate_weight.float())
        rem = start_pos % self.ratio
        self.kv[slots, rem] = kv_all
        self.score[slots, rem] = sc_all
        complete = rem == self.ratio - 1
        pooled = (self.kv[slots] * self.score[slots].softmax(dim=1)).sum(dim=1)
        # fresh request: its slots were just rewritten from group 0, so the
        # softmax only sees this request's tokens (stale rows unreachable
        # until every rem slot of the group is rewritten — guaranteed at
        # completion by construction)
        return (_reference_rms_norm(pooled.to(x.dtype), norm_weight, eps),
                complete, start_pos.long() // self.ratio)

    def seed_carry(self, slot: int, kv_row, score_row, length: int) -> None:
        if self.kv is None or length <= 0:
            return
        self.kv[slot, :length] = kv_row[:length].float()
        self.score[slot, :length] = score_row[:length].float()


class SharedIndexRuntime:
    """Model-level hand-off for the V4.1 reference indexer path.

    Owns_k layers publish (a) ``index_k_provider`` — a callable
    ``(req_keys, compress_lens) -> [B, max_len, 128]`` reading the most recent
    owner's dense reference index-K store — and (b) ``candidates`` — the
    candidate-block mask published by the candidate source layer. Non-owns_k
    index sources (layers 24/28/32/36) consume both, mirroring the official
    ``SharedAttentionRuntime.index_k`` / ``.candidates``.
    """

    def __init__(self):
        self.index_k_provider: Any = None
        self.candidates: torch.Tensor | None = None


class AscendDeepseekV41Indexer(nn.Module):
    """V4.1 indexer: fp4-quantized side attention that selects compressed positions.

    ``owns_k`` layers (kv_source layers 2/8/14/20) also write the shared index
    K cache. Non-owns_k index sources (24/28/32/36) score the shared cache;
    layers that are not index sources read the model-level topk buffer
    instead of running this module at all.
    """

    def __init__(
        self,
        vllm_config: Any,
        config: Any,
        compress_ratio: int,
        *,
        owns_k: bool,
        quant_config: Any | None = None,
        cache_config: Any,
        prefix: str = "",
    ):
        super().__init__()
        # Heavy imports stay local so this module can be syntax-checked (and
        # unit-instantiated with stubs) without the NPU stack.
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.layers.linear import ReplicatedLinear

        from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

        self.vllm_config = vllm_config
        self.config = config
        self.compress_ratio = compress_ratio
        self.owns_k = owns_k
        # The dsa_v1 bridge routes indexers on this attribute; the V4 baseline
        # indexer (deepseek_v4/indexer.py) sets it to False.
        self.is_v41_indexer = True
        self.is_candidate_source = config.candidate_source_layer_id == _layer_id_from_prefix(prefix)
        self.uses_candidates = (
            0 <= config.candidate_source_layer_id < _layer_id_from_prefix(prefix)
        )
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self.dim = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.softmax_scale = self.head_dim**-0.5
        self.eps = config.rms_norm_eps
        # Official rope for index q/k = the attention's compressed rope
        # (compress theta + YaRN); built from positions directly so no
        # metadata table layout ambiguity remains.
        # Patch 60 root cause: ``compress_rope_theta`` lives at the TOP level
        # of the flattened config; ``rope_parameters`` only carries the YaRN
        # fields (rope_type/factor/betas/original_max_position_embeddings).
        # Reading it from rope_parameters silently fell back to 40000 instead
        # of the official 160000, so both index q and k rotated with the wrong
        # base. Scores stay relative (q and k share the wrong base), so the
        # bug was invisible while top-k selected ALL visible positions
        # (total <= index_topk * ratio) and corrupted SELECTION once the cap
        # bound -- long generations degraded progressively from there.
        rp = getattr(config, "rope_parameters", {}) or {}
        self._rope_base = float(
            getattr(config, "compress_rope_theta", None)
            or rp.get("compress_rope_theta")
            or rp.get("rope_theta")
            or 160000.0
        )
        self._rope_original = int(rp.get("original_max_position_embeddings", 0) or 0)
        self._rope_factor = float(rp.get("factor", 1.0))
        self._rope_beta_fast = int(rp.get("beta_fast", 32))
        self._rope_beta_slow = int(rp.get("beta_slow", 1))
        self._inv_freq: torch.Tensor | None = None

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
            return_bias=False,
        )
        self.weights_proj = ReplicatedLinear(
            self.dim,
            self.n_heads,
            bias=False,
            quant_config=None,  # bf16 in the checkpoint
            prefix=f"{prefix}.weights_proj",
            return_bias=False,
        )

        self.k_cache = None
        self.wk = None
        self.k_norm = None
        # Official quantizes indexer q/k with per-32 fp4 (ue8m0) before
        # scoring; the reference path emulates the quant->dequant round trip.
        self.fake_quant_index_kv = True
        k_dtype = (
            torch.float8_e4m3fn
            if get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE)
            else torch.bfloat16
        )
        if owns_k:
            from vllm_ascend.models.deepseek_v4.indexer import AscendDeepseekV4IndexerCache

            self.wk = ReplicatedLinear(
                config.head_dim,
                self.head_dim,
                bias=False,
                quant_config=None,  # bf16 in the checkpoint
                prefix=f"{prefix}.wk",
                return_bias=False,
            )
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_cache = AscendDeepseekV4IndexerCache(
                head_dim=self.head_dim,
                dtype=k_dtype,
                prefix=f"{prefix}.k_cache",
                cache_config=cache_config,
                compress_ratio=self.compress_ratio,
            )
        # V4.1 keys come from the layer attention compressor latent, not from
        # an indexer-private compressor.
        self.compressor = None

        # ------------------------------------------------------------------
        # Reference pooling + dense index-K store (correctness-first path
        # until the fused NPU indexer op lands). Both are keyed by the
        # request's first KV block id (block_table[i, 0]), stable for a
        # request's lifetime.
        #
        # Ownership rules (strict; a missing row is an ERROR, never zero
        # fill — zero keys would score 0 and displace informative rows in
        # the top-k selection):
        #   * A region is owned by the request that first appended at
        #     position 0. A later append starting at position 0 means the
        #     allocator recycled the block id for a new request (or the
        #     engine re-prefills after retraction): the region is RESET
        #     before the new rows land. This is the deterministic
        #     request-end signal — the model layer sees no finish event.
        #   * ``ref_written`` is the number of rows actually appended;
        #     ``ref_slot_len`` is allocated capacity. Reads validate
        #     against ``ref_written`` only — capacity is never valid data.
        #   * Reads for an unknown slot or past the written length raise.
        #   * Reclamation is by entry count + total-row budget; only
        #     regions NOT in the last stacked batch (finished, grace
        #     window elapsed) may be reclaimed. If nothing is provably
        #     finished, we raise instead of evicting a possibly-live
        #     request (loud failure beats silent wrong answers).
        # ------------------------------------------------------------------
        self.ref_pool: ReferenceCompressorPool | None = (
            ReferenceCompressorPool(compress_ratio, config.head_dim, config.rms_norm_eps) if owns_k else None
        )
        self.ref_k: torch.Tensor | None = None  # bf16 [total_rows, index_head_dim]
        self.ref_slot_of_req: dict[int, int] = {}  # first block id -> base row
        self.ref_slot_len: dict[int, int] = {}  # first block id -> region capacity
        self.ref_written: dict[int, int] = {}  # first block id -> rows actually written
        self.ref_lru: OrderedDict[int, None] = OrderedDict()
        self.ref_total = 0
        self.ref_max_entries = 4096
        self.ref_row_budget = 4_000_000  # total rows across regions (~1 GiB bf16)
        self.ref_device: torch.device | None = None
        self._ref_op_counter = 0
        self._ref_last_touch: dict[int, int] = {}
        # Grace window for FINISHED regions only (a request absent from the
        # batch for this many store ops is treated as ended); regions of
        # requests seen in the latest stack call can never be reclaimed.
        self._ref_pin_window = 4096
        self._ref_active: set[int] = set()
        self._ref_evict_skipped = 0
        self._ref_evicted = 0
        self._ref_reset_on_reuse = 0
        # R17 diagnostic counters: reads served short of the demanded
        # compress_len (metadata-inconsistent rows, e.g. DP-padding dummies
        # whose seq_lens mirror another rank's live request) and reads with
        # no region at all. Both are logged loudly and counted instead of
        # killing the engine — the strict raise turned ONE inconsistent row
        # into EngineDeadError + 182 zero-score items (gpqa-full-r17
        # attempt 1, 2026-11-10 17:20:31; reproduced gpqa-diag-r17 with
        # needed=4905/written=138 on an idle rank).
        self._ref_short_reads = 0
        self._ref_missing_reads = 0
        # R21: appends@0 on PINNED (live/prefix-shared) regions that were
        # kept instead of reset
        self._ref_share_keeps = 0
        # R19 row-level written tracking: ``ref_written`` is max-written+1
        # and cannot prove every intermediate row was actually written.
        # ``append_ref_keys`` marks each row in a per-key bytearray; reads
        # report HOLES inside the live extent (exact key + row indices)
        # instead of silently serving stale/zero rows.
        self.ref_written_rows: dict[int, bytearray] = {}
        self._ref_hole_reads = 0

    # ------------------------------------------------------------------
    # dense reference index-K store
    # ------------------------------------------------------------------
    def _rebuild_ref_store(
        self,
        device: torch.device,
        grow_key: int | None = None,
        grow_cap: int = 0,
        drop_keys: Sequence[int] = (),
    ) -> None:
        """Reallocate the dense store, growing/dropping one region at a time.

        Regions stay contiguous per request; existing rows are preserved
        (grown regions keep their old rows, dropped regions are removed).
        """
        caps = dict(self.ref_slot_len)
        for k in drop_keys:
            caps.pop(k, None)
            self.ref_slot_of_req.pop(k, None)
            self.ref_written.pop(k, None)
            self.ref_written_rows.pop(k, None)
            self.ref_lru.pop(k, None)
            self._ref_last_touch.pop(k, None)
        if grow_key is not None:
            caps[grow_key] = grow_cap
        old_caps = dict(self.ref_slot_len)
        new_total = sum(caps.values())
        new_buf = (
            torch.zeros(new_total, self.head_dim, dtype=torch.bfloat16, device=device)
            if new_total > 0
            else None
        )
        if new_buf is not None:
            dst = 0
            for k, cap in caps.items():
                old_base = self.ref_slot_of_req.get(k)
                written = self.ref_written.get(k, 0)
                if old_base is not None and written > 0:
                    # copy only WRITTEN rows; capacity beyond the written
                    # length holds no valid data
                    keep = min(cap, written, old_caps.get(k, 0))
                    if keep > 0:
                        new_buf[dst : dst + keep].copy_(self.ref_k[old_base : old_base + keep])
                self.ref_slot_of_req[k] = dst
                dst += cap
        self.ref_slot_len = caps
        self.ref_k = new_buf
        self.ref_total = new_total
        self.ref_device = device if new_buf is not None else self.ref_device

    def _reclaim_ref_regions(self, device: torch.device) -> None:
        """Reclaim provably-finished regions when over budget.

        A region is provably finished when its key was absent from the last
        stacked batch (``_ref_active``) for longer than the grace window.
        Regions of requests still in any recent batch are never touched. If
        nothing is reclaimable we raise: silently evicting a possibly-live
        request would truncate its index-K history and corrupt selection.
        """
        while (
            len(self.ref_lru) > self.ref_max_entries
            or self.ref_total > self.ref_row_budget
        ):
            victim = None
            for cand in self.ref_lru:
                if cand in self._ref_active:
                    continue  # seen in the latest batch: possibly live
                if self._ref_last_touch.get(cand, 0) > self._ref_op_counter - self._ref_pin_window:
                    continue  # inside the finished-grace window
                victim = cand
                break
            if victim is None:
                raise RuntimeError(
                    "V4.1 reference index-K store over budget "
                    f"(entries {len(self.ref_lru)}/{self.ref_max_entries}, "
                    f"rows {self.ref_total}/{self.ref_row_budget}) but no region "
                    "is provably finished; refusing to evict possibly-live "
                    "requests (active set: "
                    f"{sorted(self._ref_active)[:8]}...)"
                )
            self._rebuild_ref_store(device, drop_keys=(victim,))
            self._ref_last_touch.pop(victim, None)
            self._ref_evicted += 1

    def _ensure_ref_region(self, req_key: int, needed: int, device: torch.device) -> None:
        self._ref_op_counter += 1
        self._ref_last_touch[req_key] = self._ref_op_counter
        cur = self.ref_slot_len.get(req_key)
        if cur is not None and cur >= needed:
            self._reclaim_ref_regions(device)
            return
        new_cap = needed if cur is None else max(needed, cur * 2)
        self._rebuild_ref_store(device, grow_key=req_key, grow_cap=new_cap)
        self.ref_lru[req_key] = None
        self._reclaim_ref_regions(device)

    def append_ref_keys(
        self,
        keys: torch.Tensor,
        req_slots: torch.Tensor,
        group_positions: torch.Tensor,
    ) -> None:
        """Scatter freshly computed index keys into the per-request regions.

        ``keys``: [n, head_dim] rows; ``req_slots``: [n] first block ids;
        ``group_positions``: [n] global compressed positions — row j lands at
        ``base(req_slots[j]) + group_positions[j]``.

        An append starting at position 0 for a region that already has
        written rows is the deterministic request-end signal: the allocator
        only recycles a first block id after its request finishes (or the
        engine re-prefills a retracted request from scratch). Either way the
        old rows belong to a dead request — the region is reset first.
        """
        if keys.numel() == 0:
            return
        gp_cpu = group_positions.detach().cpu().to(torch.int64).tolist()
        rs_cpu = req_slots.detach().cpu().to(torch.int64).tolist()
        need: dict[int, int] = {}
        min_pos: dict[int, int] = {}
        for r, g in zip(rs_cpu, gp_cpu):
            r = int(r)
            if g + 1 > need.get(r, 0):
                need[r] = g + 1
            if g < min_pos.get(r, 1 << 60):
                min_pos[r] = g
        for r, nd in need.items():
            if min_pos[r] == 0 and self.ref_written.get(r, 0) > 0:
                # R21 prefix-cache fix: with enable_prefix_caching, the SAME
                # first block backs multiple LIVE requests (identical cached
                # prefix). An append starting at 0 used to be read as "block
                # recycled -> old request dead -> reset", which destroyed the
                # in-flight sharers' index rows mid-request: their reads then
                # returned clamped live lengths and the delivered
                # cmp_sparse_indices lost whole visibility spans (caught at
                # the A-seam: L20 req0 rows 1-15 saw 1 position instead of
                # 2..16; single-op e3 proved such rows also TRAP
                # SparseFlashMla with 507015; attempt2 C12 all-wrong was the
                # silent form). The reset is NOT needed for correctness:
                # readers never pass their own written extent (live =
                # min(demand, written)) and this append rewrites the shared
                # prefix rows deterministically. Only a provably-dead region
                # (not touched within the pin window and not in the active
                # batch) is still reset for space hygiene.
                _touch = self._ref_last_touch.get(r)
                pinned = (
                    r in self._ref_active
                    or (_touch is not None
                        and _touch > self._ref_op_counter - self._ref_pin_window)
                )
                if pinned:
                    self._ref_share_keeps += 1
                    if self._ref_share_keeps <= 8 or self._ref_share_keeps % 1000 == 0:
                        import os as _os_r21

                        print(f"[V41_REFSTORE] share-keep (prefix-cache "
                              f"live region, no reset): key={r} "
                              f"old_written={self.ref_written.get(r, 0)} "
                              f"new_extent={nd} "
                              f"pid={_os_r21.getpid()} "
                              f"rank={_os_r21.environ.get('VLLM_DP_RANK', '?')} "
                              f"keeps={self._ref_share_keeps}", flush=True)
                else:
                    import os as _os_r17

                    print(f"[V41_REFSTORE] reset-on-reuse: key={r} "
                          f"old_written={self.ref_written.get(r, 0)} "
                          f"new_extent={nd} pid={_os_r17.getpid()} "
                          f"rank={_os_r17.environ.get('VLLM_DP_RANK', '?')} "
                          f"resets={self._ref_reset_on_reuse}", flush=True)
                    self._rebuild_ref_store(keys.device, drop_keys=(r,))
                    self._ref_last_touch.pop(r, None)
                    self._ref_reset_on_reuse += 1
            self._ensure_ref_region(r, nd, keys.device)
            self.ref_lru.move_to_end(r)
        base = torch.tensor([self.ref_slot_of_req[int(r)] for r in rs_cpu], dtype=torch.int64, device=keys.device)
        flat = base + group_positions.to(keys.device, dtype=torch.int64)
        if int(flat.max().item()) >= self.ref_k.shape[0] or int(flat.min().item()) < 0:
            raise RuntimeError(
                f"ref store scatter OOB: flat max {int(flat.max().item())} vs rows {self.ref_k.shape[0]}")
        # advanced-indexing assignment instead of index_copy_: the aclnn
        # index_copy path faults on this stack.
        self.ref_k[flat] = keys.to(self.ref_k.dtype)
        # R19: mark every row this append actually wrote, per key
        for r, nd in need.items():
            r = int(r)
            rows = self.ref_written_rows.setdefault(r, bytearray())
            if len(rows) < nd:
                rows.extend(bytes(nd - len(rows)))
        for r, g in zip(rs_cpu, gp_cpu):
            r = int(r)
            g = int(g)
            if g < len(self.ref_written_rows[r]):
                self.ref_written_rows[r][g] = 1
        for r, nd in need.items():
            r = int(r)
            if nd > self.ref_written.get(r, 0):
                self.ref_written[r] = nd

    def missing_rows(self, req_slot: int, upto: int) -> list[int]:
        """Row indices in [0, upto) never written for this key (R19).

        Uses the row-level written map; an empty list means every row in
        the live extent was genuinely written."""
        rows = self.ref_written_rows.get(int(req_slot))
        if rows is None:
            return list(range(max(upto, 0)))
        return [i for i in range(min(upto, len(rows))) if not rows[i]]

    def read_ref_keys(
        self, req_slot: int, compress_len: int, layer: str | None = None
    ) -> torch.Tensor:
        """Dense view of one request's index keys, up to what the store holds.

        R17 semantics change (evidence-backed): a read beyond the written
        history, or against a missing region, used to RAISE — but the demand
        comes from the batch's seq_lens, which DP-padding dummy rows mirror
        from the DP-group max while their start_pos (and therefore the rows
        the pool actually writes) is their own. ONE such inconsistent row
        raised through the piecewise region and killed the whole engine
        (EngineDeadError -> every later request 500s: gpqa-full-r17 attempt 1
        lost 182/198 items; reproduced exactly in gpqa-diag-r17 with
        needed=4905 / written=138 on an idle rank). Now the read is CAPPED
        at the written rows and logged loudly: the caller's per-row `live`
        length masks short rows out of top-k, so a dummy's garbage demand
        cannot displace real positions, and a REAL request with genuinely
        lost history stays visible (log + counters) without taking the
        service down.
        """
        req_slot = int(req_slot)
        base = self.ref_slot_of_req.get(req_slot)
        if base is not None:
            self._ref_op_counter += 1
            self._ref_last_touch[req_slot] = self._ref_op_counter
        if compress_len <= 0:
            device = self.ref_device if self.ref_device is not None else torch.device("cpu")
            return torch.zeros((0, self.head_dim), dtype=torch.bfloat16, device=device)
        if base is None:
            self._ref_missing_reads += 1
            import os as _os_r17

            print(f"[V41_REFSTORE] missing region read (capped to empty): "
                  f"key={req_slot} needed={compress_len} "
                  f"pid={_os_r17.getpid()} "
                  f"rank={_os_r17.environ.get('VLLM_DP_RANK', '?')} "
                  f"count={self._ref_missing_reads}", flush=True)
            device = self.ref_device if self.ref_device is not None else torch.device("cpu")
            return torch.zeros((0, self.head_dim), dtype=torch.bfloat16, device=device)
        written = self.ref_written.get(req_slot, 0)
        # R19 hole check: rows below the live extent that were never
        # written would be served as stale/zero data — report them by
        # exact key + layer + row indices instead of staying silent.
        live = min(compress_len, written)
        if live > 0:
            holes = self.missing_rows(req_slot, live)
            if holes:
                self._ref_hole_reads += 1
                head = ",".join(str(h) for h in holes[:8])
                print(f"[V41_REFSTORE] hole in live history: key={req_slot} "
                      f"layer={layer} rows=[{head}"
                      f"{'...' if len(holes) > 8 else ''}] hole_count={len(holes)} "
                      f"live={live} count={self._ref_hole_reads}", flush=True)
        if compress_len > written:
            self._ref_short_reads += 1
            import os as _os_r17

            print(f"[V41_REFSTORE] short read (capped): key={req_slot} "
                  f"needed={compress_len} written={written} "
                  f"pid={_os_r17.getpid()} "
                  f"rank={_os_r17.environ.get('VLLM_DP_RANK', '?')} "
                  f"count={self._ref_short_reads}", flush=True)
            return self.ref_k[base : base + written]
        return self.ref_k[base : base + compress_len]

    def stack_ref_keys(
        self, req_slots: Sequence[int], compress_lens: Sequence[int],
        layer: str | None = None,
    ) -> tuple[torch.Tensor, list[int]]:
        """[B, max_len, head_dim] dense batch view + per-request live lengths.

        Records the batch's request keys as the live set for reclamation.
        A row's live length is what the store actually holds for its key
        (R17: capped at the written rows when the demanded compress_len
        exceeds them — DP-padding dummy rows mirror another rank's seq_lens
        and legitimately hold only their own history; the caller masks
        rows whose live < demand out of top-k, so a short read never
        displaces real positions and no longer kills the engine).
        """
        B = len(req_slots)
        self._ref_active = {int(s) for s in req_slots}
        max_len = max(compress_lens, default=0)
        device = self.ref_device if self.ref_device is not None else torch.device("cpu")
        if max_len == 0:
            return torch.zeros((B, 0, self.head_dim), dtype=torch.bfloat16, device=device), [0] * B
        if self.ref_k is None or self.ref_device is None:
            # R17 capped semantics: an empty store serves empty views (the
            # caller masks short rows out of top-k) instead of raising —
            # a disabled/skipped append path must not kill the engine.
            self._ref_missing_reads += 1
            device = self.ref_device if self.ref_device is not None else torch.device("cpu")
            return (torch.zeros(len(req_slots), max_len, self.head_dim,
                                dtype=torch.bfloat16, device=device),
                    [0] * len(req_slots))
        out = torch.zeros(B, max_len, self.head_dim, dtype=self.ref_k.dtype, device=self.ref_device)
        live: list[int] = []
        for i, (slot, ln) in enumerate(zip(req_slots, compress_lens)):
            if ln <= 0:
                live.append(0)
                continue
            src = self.read_ref_keys(slot, ln, layer=layer)  # strict: raises if missing
            out[i, : src.shape[0]].copy_(src)
            live.append(src.shape[0])
        return out, live

    def _index_inv_freq(self, device) -> torch.Tensor:
        """Official precompute_freqs_cis frequencies (YaRN-blended), [lanes]."""
        if self._inv_freq is None:
            import math
            dim = self.rope_head_dim
            freqs = 1.0 / (self._rope_base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
            if self._rope_original > 0:
                def corrected_dim(rotations):
                    return dim * math.log(
                        self._rope_original / (rotations * 2 * math.pi)
                    ) / (2 * math.log(self._rope_base))
                low = max(math.floor(corrected_dim(self._rope_beta_fast)), 0)
                high = min(math.ceil(corrected_dim(self._rope_beta_slow)), dim - 1)
                ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low)
                        / max(high - low, 1e-3)).clamp(0, 1)
                smooth = 1 - ramp
                freqs = freqs / self._rope_factor * (1 - smooth) + freqs * smooth
            self._inv_freq = freqs.to(device)
        return self._inv_freq

    def _rotary_at_positions(self, lanes: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Interleaved-pair rotation of the last lanes at the given positions
        (official apply_rotary_emb math evaluated at the selected rows)."""
        inv = self._index_inv_freq(lanes.device)
        ang = positions.reshape(-1, 1).float() * inv.reshape(1, -1)
        # Match the lanes rank: [n, 1, ..., 1, lanes] so broadcasting never
        # adds a spurious dimension (2-D lanes need [n,1,lanes], 3-D with
        # heads need [n,1,1,lanes]).
        _batch = [ang.shape[0]] + [1] * (lanes.dim() - 2) + [ang.shape[1]]
        cos_f = torch.cos(ang).float().reshape(_batch)
        sin_f = torch.sin(ang).float().reshape(_batch)
        # Keep the head dims: [n, ..., rd] -> [n, ..., rd/2, 2] so the
        # [n, 1, lanes] tables broadcast over heads exactly.
        x = lanes.unflatten(-1, (lanes.shape[-1] // 2, 2))
        re, im = x[..., 0].float(), x[..., 1].float()
        out = torch.stack([re * cos_f - im * sin_f, re * sin_f + im * cos_f], dim=-1)
        return out.flatten(-2).to(lanes.dtype)

    def compute_index_key(
        self,
        latent: torch.Tensor,
        key_positions: torch.Tensor,
        quant: bool | None = None,
    ) -> torch.Tensor:
        """k = k_norm(wk(latent)) with the last rope lanes rotated (owns_k only).

        ``latent`` is this layer's RoPE-free compressed latent; a latent stands
        for the first token of its group, so group j rotates at the
        group-start TOKEN position j * ratio -- the caller passes the rope
        table rows at those positions (complex lanes, i.e. every other column
        of the duplicated cos/sin tables; [n, 1, rope_head_dim // 2]).
        """
        assert self.owns_k and self.wk is not None
        latent = latent.reshape(-1, latent.shape[-1])
        # Raw matmul + manual RMSNorm: inside the dsa_forward custom-op
        # region the wrapped Linear/RMSNorm modules mis-shape small batches
        # (5x512 @ 512x128 returned 1x128); plain torch ops stay exact and
        # match the official math.
        w = self.wk.weight.to(latent.dtype).contiguous()        # [head_dim, 512]
        k = torch.mm(latent.contiguous(), w.t().contiguous())    # [n, head_dim]
        if k.shape != (latent.shape[0], self.head_dim):
            raise RuntimeError(
                f"wk matmul shape corruption: {latent.shape} @ {w.t().shape} -> {k.shape}")
        k = k.float()
        k = k * torch.rsqrt(k.square().mean(-1, keepdim=True) + self.eps)
        k = k * self.k_norm.weight.float()
        k = k.to(torch.bfloat16)
        if k.shape != (latent.shape[0], self.head_dim):
            raise RuntimeError(f"wk norm shape corruption -> {k.shape}")
        rope_lanes = k[..., self.head_dim - self.rope_head_dim :]
        rotated = self._rotary_at_positions(rope_lanes, key_positions)
        # In-place lane write instead of torch.cat: aclnnCat rejects mixed
        # NPU memory formats (ND vs NCL), and .contiguous() preserves them.
        k = k.clone()
        k[..., self.head_dim - self.rope_head_dim :] = rotated.to(k.dtype)
        if quant is None:
            quant = self.fake_quant_index_kv
        if quant:
            fp4_fake_quant_inplace(k)
        return k

    def compute_query(
        self,
        qr: torch.Tensor,
        query_positions: torch.Tensor,
        quant: bool | None = None,
    ) -> torch.Tensor:
        qr = qr.reshape(-1, qr.shape[-1])
        _lin = self.wq_b(qr)
        _qv = _lin[0].reshape(-1, self.n_heads, self.head_dim)
        if _qv.shape[0] * _qv.shape[1] * _qv.shape[2] != _lin[0].numel():
            raise RuntimeError(
                f"wq_b output aliasing: {_lin[0].shape} -> {_qv.shape}")
        q = _qv
        rope_lanes = q[..., self.head_dim - self.rope_head_dim :]
        rotated = self._rotary_at_positions(
            rope_lanes, query_positions.reshape(-1)[: q.shape[0]])
        # In-place lane write instead of torch.cat: aclnnCat rejects mixed
        # NPU memory formats (ND vs NCL), and .contiguous() preserves them.
        q = q.clone()
        q[..., self.head_dim - self.rope_head_dim :] = rotated.to(q.dtype)
        if quant is None:
            quant = self.fake_quant_index_kv
        if quant:
            fp4_fake_quant_inplace(q)
        return q

    def score(
        self,
        hidden_states: torch.Tensor,
        query: torch.Tensor,
        index_k: torch.Tensor,
        compress_lens: torch.Tensor | int,
        query_start_loc: torch.Tensor | None = None,
        cl_backup: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dense scoring: einsum -> relu -> weighted head sum -> visibility mask.

        ``index_k``: [B, cmp_len, head_dim] dense view of the (shared) index K
        cache for the torch reference path. ``compress_lens``: how many
        compressed positions each query can see (a block becomes visible once
        the query has passed its last token) — [B, S, 1] / [S, 1] tensor or an
        int (decode). ``query`` is the flattened [T, n_heads, head_dim] form;
        pass ``query_start_loc`` to pad it into [B, S, n_heads, head_dim]
        buckets (the dsa_v1 bridge form, returning [B, S, n_positions]).
        Without it the legacy flattened [T, 1, n_heads, head_dim] view is kept.
        """
        # R29 micro-brackets: value-snapshot compress_lens after each op
        # group (same-stream clones). The R28 live capture showed cl_just
        # correct but cl_used torn to a constant — these marks name the op
        # group between which the content changes. Marks: M0 entry, M1 after
        # weights_proj, M2 after q/w scatter, M3 after einsum+relu_, M4
        # after the weighted sum (the R26 cl_used point).
        _dbgc = (
            __import__("os").environ.get("V41_CAPTURE_IDX", "")
            and torch.is_tensor(compress_lens)
        )
        if _dbgc:
            self.debug_cl_marks = [compress_lens.detach().clone()]
            self.debug_cl_ptr = compress_lens.data_ptr()
            self.debug_cl_recovered = False
        weights = self.weights_proj(hidden_states)[0] * (self.softmax_scale * self.n_heads**-0.5)
        if _dbgc:
            self.debug_cl_marks.append(compress_lens.detach().clone())
        if query_start_loc is None:
            q = query.view(hidden_states.shape[0], 1, self.n_heads, self.head_dim)
            w = weights.view(hidden_states.shape[0], 1, self.n_heads)
        else:
            B = index_k.shape[0]
            T = query.shape[0]
            # R46: replace repeat_interleave + index_put_ scatter with
            # host-computed slicing — the aicore fingerprint (VEC unit,
            # sequential MTE, CUBE clean) points at a vector-unit scatter
            # op; these two (repeat_interleave, index_put_) are the only
            # vector-unit scatter ops in the tear window (M1→M2).
            _qsl_h = query_start_loc.detach().cpu().to(torch.int64).tolist()
            _lens_h = [_qsl_h[b + 1] - _qsl_h[b]
                       for b in range(min(B, len(_qsl_h) - 1))]
            S = max(_lens_h) if _lens_h else 0
            q = query.new_zeros(B, S, self.n_heads, self.head_dim)
            w = weights.new_zeros(B, self.n_heads) \
                if S == 0 else weights.new_zeros(B, S, self.n_heads)
            if S == 0 or T == 0:
                # warmup dummy / zero-token batch — all-zero padding
                pass
            else:
                for _b in range(min(B, len(_qsl_h) - 1)):
                    _st = min(_qsl_h[_b], T)
                    _en = min(_qsl_h[_b + 1], T)
                    if _en > _st:
                        q[_b, : _en - _st] = query[_st:_en]
                        w[_b, : _en - _st] = weights[_st:_en]
        if _dbgc:
            self.debug_cl_marks.append(compress_lens.detach().clone())
        # Reference semantics:
        #   index_score = einsum("bshd,btd->bsht", q, index_k).relu_()
        #   index_score = (index_score * weights.unsqueeze(-1)).sum(dim=2)
        #   masked_fill positions >= compress_lens with -inf
        #
        # R44: replace einsum with explicit reshape+matmul — the NPU einsum
        # lowering creates internal intermediates whose shapes we can't
        # guard; 14 width crashes traced to this line. The explicit path:
        #   q (B,S,H,D) → (B, S*H, D)
        #   k (B,T,D) → transpose → (B, D, T)
        #   bmm → (B, S*H, T) → reshape (B,S,H,T)
        # Each intermediate gets a shape assertion; on failure we log and
        # return a zero-score tensor (the SEL_GUARD downstream will repair
        # the selection from host facts, so the engine survives).
        _Bq, _Sq, _Hq, _Dq = q.shape
        _Bk, _Tk, _Dk = index_k.shape
        try:
            if _Bq != _Bk or _Dq != _Dk:
                raise RuntimeError(
                    f"R44 pre-matmul: q({_Bq},{_Sq},{_Hq},{_Dq}) "
                    f"index_k({_Bk},{_Tk},{_Dk}) batch/dim mismatch")
            q_f = q.float().reshape(_Bq, _Sq * _Hq, _Dq)
            k_t = index_k.float().transpose(1, 2)
            if q_f.shape[0] != k_t.shape[0] or q_f.shape[2] != k_t.shape[1]:
                raise RuntimeError(
                    f"R44 matmul shapes: q_f{tuple(q_f.shape)} "
                    f"k_t{tuple(k_t.shape)}")
            prod = torch.bmm(q_f, k_t)
            _exp_prod = (_Bq, _Sq * _Hq, _Tk)
            if prod.shape != _exp_prod:
                raise RuntimeError(
                    f"R44 bmm output {tuple(prod.shape)} != {_exp_prod}")
            index_score = prod.reshape(_Bq, _Sq, _Hq, _Tk).relu_()
        except RuntimeError as _r44e:
            import os as _os44
            if _os44.environ.get("V41_TEAR_GUARD", "1") != "0":
                print(f"[V44_MATMUL_GUARD] pid={_os44.getpid()} "
                      f"{_r44e}", flush=True)
                # zero scores: SEL_GUARD rebuilds selection from host facts
                index_score = torch.zeros(
                    _Bq, _Sq, _Hq, max(_Tk, 1),
                    dtype=torch.float32, device=q.device)
        if _dbgc:
            self.debug_cl_marks.append(compress_lens.detach().clone())
        # weighted head sum with shape assertion
        try:
            _ws = w.unsqueeze(-1)
            if index_score.dim() != 4 or _ws.dim() != 4 or \
                    index_score.shape[:3] != _ws.shape[:3]:
                raise RuntimeError(
                    f"R44 wsum: index_score{tuple(index_score.shape)} "
                    f"w{tuple(_ws.shape)}")
            index_score = (index_score * _ws).sum(dim=2)
        except RuntimeError as _r44w:
            import os as _os44w
            print(f"[V44_WSUM_GUARD] pid={_os44w.getpid()} {_r44w}",
                  flush=True)
            index_score = torch.zeros(
                _Bq, _Sq, max(_Tk, 1),
                dtype=torch.float32, device=q.device)
        if _dbgc:
            self.debug_cl_marks.append(compress_lens.detach().clone())
        if cl_backup is not None and torch.is_tensor(compress_lens):
            # R31/R36 tear guard: the observed tear window is INSIDE this
            # function (marks M1 correct -> M2 torn = the q/w scatter
            # block). R36: visibility consumes the guarded, staggered,
            # host-minted backup DIRECTLY — a torch.where here would mint
            # an unguarded temp (the R35 crash tore exactly that temp).
            if _dbgc and not bool(
                torch.equal(compress_lens, cl_backup)
            ):
                self.debug_cl_recovered = True
            compress_lens = cl_backup
        # R25: raw scores AFTER the head-weighted sum, BEFORE any
        # masked_fill — the analyzer needs pre-filter finiteness (the
        # bridge's dbg capture snapshots this right after score returns).
        if __import__("os").environ.get("V41_CAPTURE_IDX", ""):
            self.debug_raw_score = index_score.detach()
        if __import__("os").environ.get("V41_CAPTURE_IDX", "") and torch.is_tensor(compress_lens):
            # R26 point 4: the cl actually consumed by the FIRST visibility
            # masked_fill — cloned (value snapshot) before the fill runs
            self.debug_cl_used = compress_lens.detach().clone()
        positions = torch.arange(index_k.size(1), device=index_k.device)
        if isinstance(compress_lens, int):
            visible = positions < compress_lens
            index_score = index_score.masked_fill(~visible, -torch.inf)
        else:
            cl = compress_lens if compress_lens.dim() > 1 else compress_lens.view(-1, 1)
            index_score = index_score.masked_fill(positions.view(1, 1, -1) >= cl, -torch.inf)
        return index_score

    def compute_candidates(
        self,
        index_score: torch.Tensor,
        compress_lens: torch.Tensor | int,
    ) -> torch.Tensor:
        """Level-one candidate blocks (candidate source layer only)."""
        return select_candidate_blocks(
            index_score,
            compress_lens,
            self.candidate_topk_blocks,
            self.candidate_block_size,
        )

    def select_topk(
        self,
        index_score: torch.Tensor,
        compress_lens: torch.Tensor | int,
        end_pos: int,
        offset: int,
        candidates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Candidate filtering + top-k + position-order sort, official semantics."""
        if self.is_candidate_source:
            candidates = select_candidate_blocks(
                index_score,
                compress_lens,
                self.candidate_topk_blocks,
                self.candidate_block_size,
            )
        elif self.uses_candidates and candidates is not None:
            # level two: score with our own weights, but only inside the
            # source's candidate blocks
            index_score = index_score.masked_fill(~candidates, -torch.inf)

        # top-k by score, re-sorted into position order; unreachable -> -1,
        # rest shifted by offset
        topk = min(self.index_topk, end_pos // self.compress_ratio)
        if isinstance(compress_lens, int):
            compress_lens_t = torch.tensor(compress_lens, device=index_score.device)
        elif compress_lens.dim() == 1:
            compress_lens_t = compress_lens.view(-1, 1)
        else:
            # [B, S, 1] (bridge) or [S, 1] (official prefill) already
            # broadcast against the [.., topk] index tensor
            compress_lens_t = compress_lens
        # Validity by SCORE, not just index range: top-k on a row where every
        # position is masked (-inf: invisible or dead rows) still returns
        # indices, and index-range checks alone would bless them as
        # selections. Only finite scores are selectable.
        vals, idxs = index_score.topk(topk, dim=-1, sorted=False)
        valid = torch.isfinite(vals)
        # sort indices ascending as float32: int32/int64 ArgSort has no AiCore
        # kernel (AiCpu fallback on this hot per-layer path); compressed
        # positions stay far below 2**24 so the cast is exact.
        order = idxs.to(torch.float32).argsort(dim=-1)
        idxs = idxs.gather(-1, order)
        valid = valid.gather(-1, order)
        # R37: the in_range re-check read compress_lens from device memory a
        # SECOND time inside select_topk — the R36 crash showed a tear landing
        # exactly there (req1/req2 q0: pre-finite 424/307 correct, final nnz
        # 182/65). Visibility is already enforced by score()'s masked_fill
        # (finite scores = visible positions), which is the official
        # validity rule; default OFF keeps this read out of the path.
        if __import__("os").environ.get("V41_SELECT_INRANGE", "0") == "1":
            keep = valid & (idxs < compress_lens_t)
        else:
            keep = valid
        return torch.where(keep, idxs + offset, -1).int()

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        latent: torch.Tensor | None,
        index_k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        compress_lens: torch.Tensor | int,
        end_pos: int,
        offset: int,
        candidates: torch.Tensor | None = None,
        key_cos: torch.Tensor | None = None,
        key_sin: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Reference (pure torch) forward. Returns (topk_indices, new_candidates).

        The dsa_v1 bridge drives the pieces (ref_pool / compute_index_key /
        append_ref_keys / compute_query / score / select_topk) directly so the
        persistent store stays in the loop; this forward exists so numerics
        can be diffed against the official implementation standalone.
        """
        new_key = None
        if self.owns_k and latent is not None:
            new_key = self.compute_index_key(
                latent,
                key_cos if key_cos is not None else cos,
                key_sin if key_sin is not None else sin,
            )
        query = self.compute_query(qr, cos, sin)
        index_score = self.score(hidden_states, query, index_k, compress_lens)
        topk = self.select_topk(index_score, compress_lens, end_pos, offset, candidates)
        new_candidates = None
        if self.is_candidate_source:
            new_candidates = self.compute_candidates(index_score, compress_lens)
        return topk, new_candidates


def _norm_pos_table(t: torch.Tensor, rows: int, lanes: int) -> torch.Tensor:
    """Normalize a rope position table to [rows, 1, lanes] float.

    Tables come in two layouts: complex-lane values duplicated column-wise
    (width 2*lanes; take every other column) or already one column per lane
    (width lanes). Layer groups differ; normalize instead of assuming.
    """
    t = t.reshape(rows, -1).float()
    if t.shape[-1] == 2 * lanes:
        t = t[:, 0::2]
    if t.shape[-1] != lanes:
        raise ValueError(f"position table width {t.shape[-1]} != lanes {lanes}")
    return t.unsqueeze(1)


def _apply_interleaved_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Complex-style rotation over adjacent element pairs (interleaved halves).

    Matches the official ``apply_rotary_emb``: adjacent pairs form complex
    numbers (x[2i], x[2i+1]) and rotate by the per-lane angle.
    """
    x1 = x.float().unflatten(-1, (-1, 2))
    re, im = x1[..., 0], x1[..., 1]
    # cos/sin shaped [*, lanes] from the caller
    cos_f, sin_f = cos.float(), sin.float()
    while cos_f.dim() < re.dim():
        cos_f = cos_f.unsqueeze(0)
        sin_f = sin_f.unsqueeze(0)
    return torch.stack([re * cos_f - im * sin_f, re * sin_f + im * cos_f], dim=-1).flatten(-2).to(x.dtype)


def _layer_id_from_prefix(prefix: str) -> int:
    """Extract the decoder layer index from a '...layers.N...' prefix."""
    parts = prefix.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            return int(parts[i + 1])
    raise ValueError(f"cannot find layer id in prefix {prefix!r}")


__all__ = [
    "AscendDeepseekV41Indexer",
    "ReferenceCompressorPool",
    "SharedIndexRuntime",
    "select_candidate_blocks",
    "fp4_fake_quant_inplace",
]
