# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable
from functools import lru_cache
from importlib import import_module
from typing import Any

import torch


@lru_cache
def _get_sparse_flash_mla_ops() -> tuple[Callable, Callable]:
    """Load SparseFlashMla operators without importing vllm_ascend.ops."""
    try:
        import_module("cann_ops_transformer")
        namespace = torch.ops.cann_ops_transformer
        return namespace.sparse_flash_mla, namespace.sparse_flash_mla_metadata
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "DeepSeek-V4 BF16 KV on Ascend A5 requires SparseFlashMla from a matching CANN 9.2 toolkit."
        ) from exc


# ---------------------------------------------------------------------------
# V4.1 numerics capture (debug-only; env V41_CAPTURE_SMLA=<dir>).
#
# Saves the exact op calls (kwargs + gathered cache pages) of prefill-sized
# forwards so the paged op's cmp-stream semantics can be replayed offline
# against the official reference math. V41_CAPTURE_SMLA_DECODE=1 additionally
# captures the first decode-sized calls.
# ---------------------------------------------------------------------------
_CAPTURE = {"dir": None, "n_prefill": 0, "n_decode": 0, "n_calls_decode": 0, "rank": None}


def _capture_dir() -> str | None:
    if _CAPTURE["dir"] is None:
        d = os.environ.get("V41_CAPTURE_SMLA", "")
        _CAPTURE["dir"] = d or "\0"
        if d:
            os.makedirs(d, exist_ok=True)
            _CAPTURE["rank"] = os.environ.get("VLLM_DP_RANK") or os.environ.get("LOCAL_RANK") or "0"
    return None if not _CAPTURE["dir"] or _CAPTURE["dir"] == "\0" else _CAPTURE["dir"]


def _cpu(t):
    return t.detach().cpu() if isinstance(t, torch.Tensor) else t


def _gather_pages(cache: torch.Tensor | None, block_table: torch.Tensor | None):
    """Clone only the pages a block table references (cache pools are huge)."""
    if cache is None or block_table is None or block_table.numel() == 0:
        return _cpu(cache)
    pages = block_table.detach().reshape(-1).unique().tolist()
    idx = torch.tensor(pages, dtype=torch.long, device=cache.device)
    return {"pages_idx": pages, "rows": cache.detach().index_select(0, idx).cpu()}


def _capture_call(tag: str, q: torch.Tensor | None, kwargs: dict[str, Any], out=None) -> None:
    try:
        _capture_call_inner(tag, q, kwargs, out)
    except Exception:  # noqa: BLE001 - debug hook must never kill serving
        pass


def _capture_call_inner(tag: str, q: torch.Tensor | None, kwargs: dict[str, Any], out=None) -> None:
    d = _capture_dir()
    if d is None:
        return
    T = q.shape[0] if q is not None else 0
    is_decode = T < 16
    if is_decode:
        # skip engine warmups: only capture decodes after a real prefill
        if (
            os.environ.get("V41_CAPTURE_SMLA_DECODE", "0") != "1"
            or _CAPTURE["n_prefill"] == 0
            or _CAPTURE["n_decode"] >= int(os.environ.get("V41_CAPTURE_DECODE_CAP", "60") or 60)
        ):
            return
        # select decode steps by call index: each decode step issues one call
        # per layer in order, so n_calls % n_layers identifies the layer.
        _CAPTURE["n_calls_decode"] += 1
        n_layers = int(os.environ.get("V41_CAPTURE_DECODE_LAYERS_N", "40") or 40)
        keep_layers = {
            int(x) for x in os.environ.get("V41_CAPTURE_DECODE_LAYERS", "0,2,20").split(",") if x.strip()
        }
        layer_in_step = (_CAPTURE["n_calls_decode"] - 1) % n_layers
        if layer_in_step not in keep_layers:
            return
        # skip the first FROM decode CALLS (warmups + early steps)
        skip_from = int(os.environ.get("V41_CAPTURE_DECODE_FROM", "0") or 0)
        if _CAPTURE["n_calls_decode"] <= skip_from:
            return
        # patch 60: subsample by decode STEP (one step = n_layers calls) so a
        # wide generation range is covered without filling the disk
        step_mod = int(os.environ.get("V41_CAPTURE_DECODE_MOD", "1") or 1)
        if step_mod > 1 and ((_CAPTURE["n_calls_decode"] - 1) // n_layers) % step_mod != 0:
            return
        _CAPTURE["n_decode"] += 1
        name = f"decode_{_CAPTURE['n_calls_decode']:05d}_L{layer_in_step}"
    else:
        # skip the profiling/warmup forward (max_num_batched_tokens sized)
        if not (3 < T < 1000) or _CAPTURE["n_prefill"] >= 60:
            return
        _CAPTURE["n_prefill"] += 1
        name = f"prefill_{_CAPTURE['n_prefill']:03d}"
    try:
        import torch.npu  # noqa: F401

        torch.npu.synchronize()
    except Exception:  # noqa: BLE001
        pass
    small = {}
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            if v.numel() <= 4096:
                small[k] = _cpu(v)
        elif v is None or isinstance(v, (int, float, str, bool)):
            small[k] = v
    payload = {
        "tag": tag,
        "T": T,
        "small": small,
        "q": _cpu(q),
        "ori_kv_pages": _gather_pages(kwargs.get("ori_kv"), kwargs.get("ori_block_table")),
        "cmp_kv_pages": _gather_pages(kwargs.get("cmp_kv"), kwargs.get("cmp_block_table")),
        "metadata": _cpu(kwargs.get("metadata")),
        "cmp_sparse_indices": _cpu(kwargs.get("cmp_sparse_indices")),
        "sinks": _cpu(kwargs.get("sinks")),
        "op_out": _cpu(out[0]) if isinstance(out, (tuple, list)) else _cpu(out),
    }
    try:
        torch.save(payload, os.path.join(d, f"p{os.getpid()}_{name}_{tag}.pt"))
    except Exception:  # noqa: BLE001 - debug hook must never kill serving
        pass


def _capture_metadata(kwargs: dict[str, Any]) -> None:
    d = _capture_dir()
    if d is None or _CAPTURE["n_prefill"] + _CAPTURE["n_decode"] > 120:
        return
    try:
        payload = {
            k: (_cpu(v) if isinstance(v, torch.Tensor) and v.numel() <= 4096 else v)
            for k, v in kwargs.items()
        }
        torch.save(payload, os.path.join(d, f"r{_CAPTURE['rank']}_md_{_CAPTURE['n_prefill']:03d}.pt"))
    except Exception:  # noqa: BLE001
        pass



def _add_compressed_kv_lengths(kwargs: dict[str, Any]) -> None:
    # Pure SWA layers must carry NO compressed lengths — the aclnn op rejects
    # cmp_residual_kv without a cmp stream. The metadata op signals this with
    # an explicit has_cmp_kv kwarg; the attention op with a cmp_kv cache arg.
    if "has_cmp_kv" in kwargs:
        has_cmp = bool(kwargs["has_cmp_kv"])
    else:
        has_cmp = "cmp_kv" in kwargs
    if not has_cmp:
        for k in ("seqused_cmp_kv", "cmp_residual_kv", "max_seqlen_cmp_kv"):
            kwargs.pop(k, None)
        return
    cmp_ratio = kwargs.get("cmp_ratio") or 0
    seqused_ori_kv = kwargs.get("seqused_ori_kv")
    # cmp_ratio == 1 is valid (DeepSeek-V4.1 ratio-1 layers: every token is a
    # compressed position, cmp lengths equal the ori lengths).
    if cmp_ratio < 1 or seqused_ori_kv is None:
        return
    kwargs.setdefault("seqused_cmp_kv", seqused_ori_kv // cmp_ratio)
    kwargs.setdefault("cmp_residual_kv", seqused_ori_kv % cmp_ratio)
    if kwargs.get("max_seqlen_cmp_kv") is None and kwargs.get("max_seqlen_ori_kv") is not None:
        kwargs["max_seqlen_cmp_kv"] = kwargs["max_seqlen_ori_kv"] // cmp_ratio


def sparse_flash_mla_metadata(**kwargs):
    """Adapt existing DSA metadata kwargs to SparseFlashMla BF16 KV."""
    kwargs.pop("device", None)
    kwargs.pop("kv_quant_mode", None)
    # This adapter is only selected for the BF16 paged-KV path. SparseFlashMla
    # accepts PA_BBND for this cache; PA_ND belongs to the FP8 quantized op.
    kwargs["layout_kv"] = "PA_BBND"
    if "seqused_kv" in kwargs:
        kwargs["seqused_ori_kv"] = kwargs.pop("seqused_kv")
    if "max_seqlen_kv" in kwargs:
        kwargs["max_seqlen_ori_kv"] = kwargs.pop("max_seqlen_kv")
    _add_compressed_kv_lengths(kwargs)
    _capture_metadata(kwargs)
    _, metadata_op = _get_sparse_flash_mla_ops()
    return metadata_op(**kwargs)


def sparse_flash_mla(q: torch.Tensor, **kwargs):
    """Adapt existing DSA attention kwargs to SparseFlashMla BF16 KV."""
    kwargs.pop("kv_quant_mode", None)
    kwargs.pop("tile_size", None)
    kwargs.pop("rope_head_dim", None)
    kwargs["layout_kv"] = "PA_BBND"
    if "seqused_kv" in kwargs:
        kwargs["seqused_ori_kv"] = kwargs.pop("seqused_kv")
    # Paged KV (PA_BBND) addresses the caches through block tables + seqused
    # lengths; the aclnn op rejects KV cu_seqlens in this mode (TND only).
    kwargs.pop("cu_seqlens_ori_kv", None)
    kwargs.pop("cu_seqlens_cmp_kv", None)
    # aclnnSparseFlashMla requires cmp_ratio in [1, 128] even for pure SWA
    # layers (has_cmp_kv=False); the value is inert there, so clamp 0 -> 1.
    if int(kwargs.get("cmp_ratio") or 0) < 1:
        kwargs["cmp_ratio"] = 1
    _add_compressed_kv_lengths(kwargs)
    attention_op, _ = _get_sparse_flash_mla_ops()
    out = attention_op(q, **kwargs)
    _capture_call("smla", q, kwargs, out=out)
    return out
