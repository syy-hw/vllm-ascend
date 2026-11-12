import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import torch
import torch.distributed as dist
import torch_npu
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImplBase,
    AttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.dsa_attn_kv_plan import (
    get_dsa_attn_kv_plan,
    is_a5_bf16_kv_enabled,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    enable_pcp,
    get_or_register_attention_buffer,
    maybe_save_kv_layer_to_connector,
    notify_kv_cache_written,
    split_decodes_and_prefills,
    wait_for_kv_layer_from_connector,
)
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import record_attention_compute_start
from vllm_ascend.distributed.parallel_state import get_otp_group
from vllm_ascend.models.deepseek_v4.compressor import AscendCompressorMetadata
from vllm_ascend.models.deepseek_v4.indexer import AscendIndexerMetadata, IndexerOverlapPlan
from vllm_ascend.ops.cv_linear import CVLinearWrapper
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa, get_full_cos_and_sin_dsa
from vllm_ascend.quantization.methods import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import (
    get_potential_max_tokens,
    npu_stream_switch,
    olora_tp_enable,
    oproj_tp_enable,
)
from vllm_ascend.worker.device_metadata import (
    DeviceMetadataStage,
    DeviceMetadataTask,
    wait_for_device_metadata,
)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_ascend.ops.triton.rms_norm import triton_q_rms

if HAS_TRITON:
    from vllm_ascend.ops.triton.rms_norm import triton_q_rms  # noqa: F811
else:
    triton_q_rms = None  # type: ignore


# The SAS and QLI metadata operators use a fixed 1024-element int32 layout.
DSA_METADATA_BUFFER_SIZE = 1024
CompressorMetadataOutput: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

_DSV4_DSA_OVERLAP_STREAM = None


def build_compressor_metadata_out(
    metadata: Any,
    compress_ratio: int,
    outputs: CompressorMetadataOutput,
    vllm_config: VllmConfig,
) -> None:
    assert metadata.full_compress_cos is not None
    assert metadata.full_compress_sin is not None
    assert metadata.start_pos is not None
    assert metadata.num_actual_reqs is not None
    full_compress_cos = metadata.full_compress_cos.view(
        metadata.full_compress_cos.shape[0],
        metadata.full_compress_cos.shape[-1],
    )
    full_compress_sin = metadata.full_compress_sin.view(
        metadata.full_compress_sin.shape[0],
        metadata.full_compress_sin.shape[-1],
    )
    torch.ops._C_ascend.compressor_metadata_out(
        full_compress_cos,
        full_compress_sin,
        metadata.query_start_loc,
        metadata.start_pos,
        metadata.block_table,
        metadata.storage_block_size,
        get_dsa_attn_kv_plan(vllm_config).get_dsa_compressor_slot_mapping_format(),
        compress_ratio,
        metadata.num_actual_reqs,
        *outputs,
    )


def dsv4_dsa_overlap_stream() -> torch.npu.Stream:
    global _DSV4_DSA_OVERLAP_STREAM
    if _DSV4_DSA_OVERLAP_STREAM is None:
        _DSV4_DSA_OVERLAP_STREAM = torch_npu.npu.Stream()
    return _DSV4_DSA_OVERLAP_STREAM


# ---------------------------------------------------------------------------
# V4.1 decode debugging capture (patch 60; pure debug tooling, never serving).
#   V41_CAPTURE_IDX=<dir>   indexer bridge internals per decode step
#   V41_DUMP_DECODE=<dir>   per-layer attention in/out hidden per decode step
# Both sample by TOTAL context length (FROM/TO/STEP envs) so a long generation
# is covered sparsely; files carry layer id and total in their names.
# ---------------------------------------------------------------------------
_V41_DBG: dict = {"idx": None, "dump": None}


def _v41_dbg_enabled(which: str) -> bool:
    import os

    if _V41_DBG[which] is None:
        d = os.environ.get("V41_CAPTURE_IDX" if which == "idx" else "V41_DUMP_DECODE", "")
        _V41_DBG[which] = bool(d)
        if d:
            import os as _os

            _os.makedirs(d, exist_ok=True)
    return bool(_V41_DBG[which])


def _v41_dbg_total_ok(total: int) -> bool:
    import os

    frm = int(os.environ.get("V41_DBG_FROM", "0") or 0)
    to = int(os.environ.get("V41_DBG_TO", "1000000000") or 1000000000)
    step = int(os.environ.get("V41_DBG_STEP", "1") or 1)
    # Patch 70: FROM-relative grid. The absolute ``total % step`` grid can never
    # coincide with a reference decoder that records at totals prompt_len + 16k
    # (e.g. 185 + 16k is odd); anchoring on FROM makes both samplers hit the
    # same totals so layer dumps and reference records align one-to-one.
    return frm <= total <= to and (step <= 1 or (total - frm) % step == 0)


def _v41_dbg_layer_ok(layer_name: str, env: str = "V41_DBG_LAYERS") -> bool:
    import os
    import re

    m = re.search(r"layers\.(\d+)\.", layer_name)
    lid = int(m.group(1)) if m else -1
    keep = os.environ.get(env, "")
    if keep.strip():
        ids = {x.strip() for x in keep.split(",")}
        if str(lid) not in ids:
            return False
    return True


def _v41_dbg_path(dirname_env: str, layer_name: str, total: int, suffix: str) -> str:
    import os
    import re

    d = os.environ.get(dirname_env, ".")
    m = re.search(r"layers\.(\d+)\.", layer_name)
    lid = m.group(1) if m else "x"
    rank = os.environ.get("VLLM_DP_RANK") or os.environ.get("LOCAL_RANK") or "0"
    # R23: unique per (pid, layer) counter — cap files must NEVER overwrite
    # (last-writer-wins hid every call but the final one)
    key = (os.getpid(), lid, suffix)
    _V41_DBG.setdefault("n", {})
    _V41_DBG["n"][key] = _V41_DBG["n"].get(key, 0) + 1
    return os.path.join(
        d, f"p{os.getpid()}_r{rank}_L{lid}_t{total}_"
           f"{_V41_DBG['n'][key]:05d}_{suffix}.pt")


def _v41_dbg_save(payload: dict, path: str) -> None:
    import torch as _torch

    _torch.save(payload, path)


# ---------------------------------------------------------------------------
# R21 A/B index seam capture (env V41_AB_CAPTURE=<dir>, layers via
# V41_AB_LAYERS="20,21,24"). Two snapshots with identical identity fields:
#   A — the selection the indexer bridge DELIVERS (values cloned at return)
#   B — the cmp_sparse_indices the attention call actually RECEIVES
# Files are append-only: p{pid}_r{rank}_L{lid}_{side}{n:05d}_T{T}.pt, the
# per-(pid,layer,side) counter never resets, so nothing is overwritten.
# ---------------------------------------------------------------------------

_V41_AB: dict = {"dir": None, "n": {}}


def _v41_ab_enabled() -> bool:
    import os

    if _V41_AB["dir"] is None:
        d = os.environ.get("V41_AB_CAPTURE", "")
        _V41_AB["dir"] = d or "\0"
        if d:
            os.makedirs(d, exist_ok=True)
    return _V41_AB["dir"] not in ("\0", "")


def _v41_ab_layer_ok(layer_name: str) -> bool:
    import os
    import re

    m = re.search(r"layers\.(\d+)\.", layer_name)
    lid = int(m.group(1)) if m else -1
    keep = os.environ.get("V41_AB_LAYERS", "")
    if keep.strip():
        return str(lid) in {x.strip() for x in keep.split(",")}
    return True


def _v41_ab_capture(
    side: str,
    layer_name: str,
    tensor,
    seq_lens_cpu,
    start_pos_cpu,
    first_blocks,
    qsl_cpu,
    pool=None,
) -> None:
    """Snapshot the exact values crossing the seam; clone() BEFORE anything
    else can mutate the source buffer. Prefill-scale only (T >= 32) so
    decode steps cannot flood the dir. `pool` (B side): watchpoint on the
    shared candidates buffer at THIS layer's attention entry — value
    snapshot + storage identity, so a mid-forward write can be bracketed to
    a single layer gap."""
    import os
    import re

    try:
        if tensor is None or not _v41_ab_enabled():
            return
        flat = tensor.detach().reshape(tensor.shape[0], -1)
        T = int(flat.shape[0])
        if T < 32 or not _v41_ab_layer_ok(layer_name):
            return
        m = re.search(r"layers\.(\d+)\.", layer_name)
        lid = m.group(1) if m else "x"
        pid = os.getpid()
        key = (pid, lid, side)
        _V41_AB["n"][key] = _V41_AB["n"].get(key, 0) + 1
        payload = {
            "side": side, "layer": layer_name, "lid": lid, "pid": pid,
            "rank": os.environ.get("VLLM_DP_RANK")
            or os.environ.get("LOCAL_RANK") or "0",
            "seq": _V41_AB["n"][key], "T": T,
            # value snapshot first — later device writes cannot change it
            "values": flat.clone().cpu(),
            "seq_lens": [int(x) for x in seq_lens_cpu],
            "start_pos": [int(x) for x in start_pos_cpu],
            "first_blocks": [int(x) for x in first_blocks],
            "qsl": [int(x) for x in qsl_cpu],
        }
        if pool is not None and pool.numel() <= (1 << 22):
            payload["pool_vals"] = pool.detach().clone().cpu()
            payload["pool_id"] = {
                "storage_ptr": pool.storage().data_ptr(),
                "data_ptr": pool.data_ptr(),
                "offset": pool.storage_offset(),
                "stride": list(pool.stride()),
                "shape": list(pool.shape),
            }
        import torch as _t

        _t.save(payload, os.path.join(
            _V41_AB["dir"],
            f"p{pid}_r{payload['rank']}_L{lid}_{side}{payload['seq']:05d}_T{T}.pt"))
    except Exception as _e:  # noqa: BLE001 — debug hook must never kill serving
        import os as _o
        if _o.environ.get("V41_AB_LOUD", ""):
            import traceback as _tb
            print(f"[V41_AB_ERR] {type(_e).__name__}: {_e} "
                  f"{_tb.format_exc().splitlines()[-1]}", flush=True)


def _v41_layer_total(req_metadata) -> int:
    n = req_metadata.num_actual_reqs
    n = int(n) if n is not None else req_metadata.seq_lens.shape[0]
    if n == 0:
        return 0
    return int(req_metadata.seq_lens[:n].max().item())


def _is_w8a8_dynamic(linear) -> bool:
    """True iff ``linear`` is wired up with ``AscendW8A8DynamicLinearMethod``."""
    quant_method = getattr(linear, "quant_method", None)
    if quant_method is None or isinstance(quant_method, AscendUnquantizedLinearMethod):
        return False
    inner_method = getattr(quant_method, "quant_method", None)
    return isinstance(inner_method, AscendW8A8DynamicLinearMethod)


def _has_weight_scale(linear) -> bool:
    return getattr(linear, "weight_scale", None) is not None


def _dsa_layout_kv(vllm_config: VllmConfig) -> str:
    return get_dsa_attn_kv_plan(vllm_config).layout_kv


def _dsa_swa_only_cmp_ratio(compress_ratio: int, vllm_config: VllmConfig, has_compressed: bool = False) -> int:
    """BF16 SWA-only attention takes no compressed stream; otherwise keep main's value."""
    if is_a5_bf16_kv_enabled(vllm_config) and compress_ratio <= 1 and not has_compressed:
        return 0
    return max(compress_ratio, 1)


def _dsa_uses_indexer(compress_ratio: int, is_compressed_layer: bool) -> bool:
    """Compressed layers that run a lightning indexer: V4 ratio 4, V4.1 ratio 1/2/4.

    C128 compressed layers (V4) never own an indexer, so they are excluded.
    """
    return is_compressed_layer and compress_ratio != 128


class AscendDSABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "ASCEND_DSA"

    @staticmethod
    def get_builder_cls():
        from vllm_ascend.utils import enable_dsa_cp

        use_dsa_cp = enable_dsa_cp()
        use_pcp = enable_pcp()
        if use_dsa_cp and use_pcp:
            raise ValueError("Legacy DSACP and PCP cannot be enabled at the same time.")
        if use_dsa_cp:
            from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPMetadataBuilder

            return AscendDSACPMetadataBuilder
        if use_pcp:
            from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSAPCPMetadataBuilder

            return AscendDSAPCPMetadataBuilder
        return AscendDSAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        return num_blocks, block_size, num_kv_heads, head_size

    @staticmethod
    def get_scale_shape(num_blocks: int, block_size: int, scale_size: int) -> tuple[int, ...]:
        return num_blocks, block_size, scale_size

    @staticmethod
    def get_impl_cls() -> type[AttentionImplBase[Any]]:
        from vllm_ascend.utils import enable_dsa_cp

        use_dsa_cp = enable_dsa_cp()
        use_pcp = enable_pcp()
        if use_dsa_cp and use_pcp:
            raise ValueError("Legacy DSACP and PCP cannot be enabled at the same time.")
        if use_dsa_cp:
            from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPImpl

            return AscendDSACPImpl
        if use_pcp:
            from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSAPCPImpl

            return AscendDSAPCPImpl
        return AscendDSAImpl

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [2, 4, 8, 16, 32, 64, 128]


class AscendDSAC4Backend(AscendDSABackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND_DSA_C4"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # Align with upstream's logical block-size contract: Ascend's physical
        # 32/64/128-token C4 pages represent 128/256/512 raw scheduler tokens.
        return [128, 256, 512]


class AscendDSAC128Backend(AscendDSABackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND_DSA_C128"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # Align with upstream's logical block-size contract: Ascend's physical
        # 32/64/128-token C128 pages represent 4096/8192/16384 raw scheduler tokens.
        return [4096, 8192, 16384]


class AscendDSASWABackend(AscendDSABackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND_DSA_SWA"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [32, 64, 128]


class AscendDSAC4StateBackend(AscendDSABackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND_DSA_C4_STATE"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [2, 4, 8]


class AscendDSAC128StateBackend(AscendDSABackend):
    @staticmethod
    def get_name() -> str:
        return "ASCEND_DSA_C128_STATE"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        if get_current_hardware_profile().supports(HardwareCapability.DSA_C128_STATE_SMALL_BLOCK_SIZES):
            return [4, 8, 16]
        return [8, 16, 32]


@dataclass
class AscendDSAReqMetadata:
    """Unified metadata for all requests in one attention invocation."""

    block_table: torch.Tensor
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor | None
    storage_block_size: int
    query_start_loc: torch.Tensor

    num_compressed_tokens: int | None = None
    sin: torch.Tensor = None
    cos: torch.Tensor = None
    full_compress_sin: torch.Tensor = None
    full_compress_cos: torch.Tensor = None
    start_pos: torch.Tensor | None = None
    num_actual_reqs: int | None = None
    sas_metadata: torch.Tensor = None
    qli_metadata: torch.Tensor = None
    compressor_metadata: CompressorMetadataOutput | None = None
    compressor_metadata_group_id: int | None = None
    attn_mask: torch.Tensor | None = None
    cu_cmp_seqlen_list: torch.Tensor = None
    ori_win_left: int | None = None
    ori_win_right: int | None = None
    dspark_swa_indices: torch.Tensor | None = None
    # V4.1 graph-decode freshness channel: the SAME dict object is passed on
    # every build (like sas_metadata: a builder-owned persistent buffer set).
    # Captured kernels bind these tensors' storage at capture; each eager
    # build() rewrites the contents, so graph replay reads THIS step's values.
    v41_persist: dict | None = None


# Registered persist sets (one per DSA attention group). The eager runner
# hook (v2 model_runner prepare_inputs) publishes per-step engine request
# slot ids into every registered set: the stable request identity for the
# device-side pool state (first block ids collide under shared prefixes).
_V41_PERSIST_REGISTRY: list[dict] = []

# Layer-IO capture (V41_LAYER_IO_DIR=<dir>, first-divergence experiment):
# per attention layer, the attention OUTPUT rows of every executed forward
# land in a persistent per-layer buffer the captured graph REWRITES (pure
# device ops — replay-safe; the in-graph step counter proves the file for
# step N is written only AFTER step N actually executed). Eager steps write
# files directly at forward time; graph steps are drained by the NEXT
# build_req_metadata (post-replay). "seq" carries seqused[0] so the
# comparison tool can filter decode steps of the TARGET request apart from
# warmup/capture dummies (seq==1) and helper requests.
_V41_LAYER_IO_DIR: str | None = None


def _v41_layer_io_dir() -> str | None:
    global _V41_LAYER_IO_DIR
    if _V41_LAYER_IO_DIR is None:
        _V41_LAYER_IO_DIR = __import__("os").environ.get(
            "V41_LAYER_IO_DIR", "") or "\0"
    return None if _V41_LAYER_IO_DIR == "\0" else _V41_LAYER_IO_DIR


# Real-step gate for the post-replay drain (R17 lite identity record),
# CONTENT-BASED (live evidence 2026-11-10 15:44: a host flag set in the
# prepare hook did NOT stop idle-tick dummies from draining — dummy ticks
# replay without the flag being refreshed). At drain time the entry itself
# is the truth: a DP idle-tick dummy runs a 1-token batch at position 0
# (ntok==1, seq==0), while real decode advances position (seq>=2) and real
# prefill carries ntok>1. One 8-byte D2H per entry — negligible next to a
# replay. Without this gate dummy replays outnumber real steps and flood
# the per-request record (graph_el5 io dir reached ~6M files).
V41_IO_DUMMY_NTOK = 1
V41_IO_DUMMY_SEQ = 1


def _v41_io_ent_is_dummy(ent) -> bool:
    try:
        return (int(ent["ntok"].detach().cpu().item()) <= V41_IO_DUMMY_NTOK
                and int(ent["seq"].detach().cpu().item()) <= V41_IO_DUMMY_SEQ)
    except Exception:  # noqa: BLE001 - never kill serving on debug paths
        return False


def _v41_io_drain_dir() -> str | None:
    """Drain directory: the attention layer-IO dir when set, else the lite
    engram identity dir (V41_ENG_IO_DIR). Unset everywhere -> drain off."""
    d = _v41_layer_io_dir()
    if d is not None:
        return d
    return __import__("os").environ.get("V41_ENG_IO_DIR", "") or None


def _v41_io_write_file(io_dir: str, layer_name: str, ent: dict) -> None:
    import os

    try:
        os.makedirs(io_dir, exist_ok=True)
        step = int(ent["step"].detach().cpu().item())
        ntok = int(ent["ntok"].detach().cpu().item())
        seq = int(ent["seq"].detach().cpu().item())
        rank = os.environ.get("VLLM_DP_RANK", os.environ.get("RANK", "0"))
        safe = layer_name.replace(".", "_").replace("#", "-")
        def _row(k):
            v = ent.get(k)
            if v is None:
                return None
            return v[: min(ntok, 2) if k in ("out", "inp", "q") else v.shape[0]].detach().cpu()
        torch.save(
            {"layer": layer_name, "step": step, "ntok": ntok, "seq": seq,
             "out": _row("out"), "inp": _row("inp"), "q": _row("q"),
             "slot": ent.get("slot", None) if not isinstance(ent.get("slot"), torch.Tensor) else ent["slot"].detach().cpu(),
             "krow": ent["krow"][:2].detach().cpu()
             if "krow" in ent else None,
             "obt": ent["obt"].detach().cpu() if "obt" in ent else None,
             "sq4": ent["sq4"].detach().cpu() if "sq4" in ent else None,
             "sas": ent["sas"].detach().cpu() if "sas" in ent else None,
             "sel": ent["sel"].detach().cpu() if "sel" in ent else None,
             "sinks": ent["sinks"].detach().cpu() if "sinks" in ent else None,
             "ori_rows": ent["ori_rows"].detach().cpu()
             if "ori_rows" in ent else None,
             "kvsum": ent["kvsum"].detach().cpu()
             if "kvsum" in ent else None,
             "wrow": ent["wrow"].detach().cpu()
             if "wrow" in ent else None,
             "wslot": ent["wslot"].detach().cpu()
             if "wslot" in ent else None,
             "wrow_pre": ent["wrow_pre"].detach().cpu()
             if "wrow_pre" in ent else None,
             "ctl_row": ent["ctl_row"].detach().cpu()
             if "ctl_row" in ent else None,
             "kvpre": ent["kvpre"].detach().cpu()
             if "kvpre" in ent else None,
             "kvrot": ent["kvrot"].detach().cpu()
             if "kvrot" in ent else None,
             "kvpost": ent["kvpost"].detach().cpu()
             if "kvpost" in ent else None,
             "qamax": ent["qamax"].detach().cpu()
             if "qamax" in ent else None,
             "qdiv": ent["qdiv"].detach().cpu()
             if "qdiv" in ent else None,
             "qexp": ent["qexp"].detach().cpu()
             if "qexp" in ent else None,
             "qscale": ent["qscale"].detach().cpu()
             if "qscale" in ent else None,
             "qq": ent["qq"].detach().cpu()
             if "qq" in ent else None,
             "qout": ent["qout"].detach().cpu()
             if "qout" in ent else None,
             # engram-site io entries (model.py _v41_eng_io_touch): rows
             # this execution READ + pre/post-apply hidden + positions
             "pos": ent["pos"].detach().cpu() if "pos" in ent else None,
             "engread": ent["engread"].detach().cpu()
             if "engread" in ent else None,
             "engpre": ent["engpre"].detach().cpu()
             if "engpre" in ent else None,
             "engpost": ent["engpost"].detach().cpu()
             if "engpost" in ent else None},
            os.path.join(io_dir, f"r{rank}_p{os.getpid()}_s{step:06d}_{safe}.pt"))
    except Exception as exc:  # noqa: BLE001 - debug hook must not kill serving
        try:
            with open(f"{io_dir}/WRITE_FAILED_{layer_name.replace('/', '_')}.txt",
                      "a") as fh:
                fh.write(repr(exc) + "\n")
        except Exception:  # noqa: BLE001
            pass


def v41_io_capture(io, layer_name, out, hidden_states, q, swa_kv_cache,
                   slot_mapping, attn_kwargs, kvprobe=None):
    """One layer-IO capture step (first-divergence instrumentation).

    Pure device ops so the FULL graph can RECORD them: every replay rewrites
    the entry buffers with THAT step's values. Two hard rules learned from
    failed cycles: (1) index caches with 1-element 1-D int64 tensors only —
    0-dim scalar indices route through LocalScalarDenseNpu and force a
    stream synchronize, illegal during capture (EE1016/107027, cycle56d);
    (2) record the WHOLE (block, offset) slot pair — copying only `_t`
    elements (_t==1 on decode) left the offset half at its init 0 and made
    every recorded pair look like (block, 0) (cycle55c/56c lesson).
    Returns (key, entry); the caller registers/drains.
    """
    _t0 = min(int(out.shape[0]), 2)
    _or = out[:_t0].reshape(_t0, -1)
    _ir = hidden_states[:_t0].reshape(_t0, -1)
    _qr = q[:_t0].reshape(_t0, -1)
    _key = (f"{layer_name}#w{_or.shape[-1]}x{_ir.shape[-1]}"
            f"x{_qr.shape[-1]}")
    _ent = io.get(_key)
    _rpp = int(swa_kv_cache.shape[1])
    if _ent is None:
        _snk = attn_kwargs.get("sinks")
        _snk_w = int(_snk.reshape(-1).numel()) if _snk is not None else 1
        _ent = {
            "step": torch.zeros(1, dtype=torch.int64, device=out.device),
            "ntok": torch.zeros(1, dtype=torch.int64, device=out.device),
            "seq": torch.zeros(1, dtype=torch.int64, device=out.device),
            "out": torch.zeros(2, _or.shape[-1], dtype=out.dtype,
                               device=out.device),
            "inp": torch.zeros(2, _ir.shape[-1], dtype=hidden_states.dtype,
                               device=out.device),
            "q": torch.zeros(2, _qr.shape[-1], dtype=q.dtype,
                             device=out.device),
            "slot": torch.zeros(4, dtype=torch.int64, device=out.device),
            "krow": torch.zeros(2, swa_kv_cache.shape[-1],
                                dtype=swa_kv_cache.dtype, device=out.device),
            "obt": torch.zeros(4, dtype=torch.int64, device=out.device),
            "sq4": torch.zeros(4, dtype=torch.int64, device=out.device),
            "sas": torch.zeros(64, dtype=torch.int32, device=out.device),
            "sel": torch.full((8,), -7, dtype=torch.int32,
                              device=out.device),
            "sinks": torch.zeros(2, max(_snk_w, 1), dtype=out.dtype,
                                 device=out.device),
            "ori_rows": torch.zeros(128, swa_kv_cache.shape[-1],
                                    dtype=swa_kv_cache.dtype,
                                    device=out.device),
        }
        if kvprobe is not None:
            for _pn in ("pre", "rot", "post"):
                _p = kvprobe.get(_pn)
                if _p is not None:
                    _ent[f"kv{_pn}"] = torch.zeros_like(_p)
            for _pn in ("qamax", "qdiv", "qexp", "qscale", "qq", "qout"):
                _p = kvprobe.get(_pn)
                if _p is not None:
                    _ent[_pn] = torch.zeros_like(_p)
        io[_key] = _ent
    _ent["step"] += 1
    _t = min(int(out.shape[0]), 2)
    _ent["ntok"].fill_(int(out.shape[0]))
    _sq = attn_kwargs.get("seqused_kv")
    if _sq is not None and _sq.numel() >= 1:
        _ent["seq"][0] = _sq.reshape(-1)[0].long()
    _ent["out"][:_t] = _or
    _ent["inp"][:_t] = _ir
    _ent["q"][:_t] = _qr
    if kvprobe is not None:
        for _pn in ("pre", "rot", "post"):
            _p = kvprobe.get(_pn)
            _e = _ent.get(f"kv{_pn}")
            if _p is not None and _e is not None and _p.shape == _e.shape:
                _e.copy_(_p)
        for _pn in ("qamax", "qdiv", "qexp", "qscale", "qq", "qout"):
            _p = kvprobe.get(_pn)
            _e = _ent.get(_pn)
            if _p is not None and _e is not None and _p.shape == _e.shape:
                _e.copy_(_p)
    if slot_mapping is not None and slot_mapping.numel() >= 1:
        _flat = slot_mapping.reshape(-1)
        _nsl = min(4, int(_flat.numel()), _ent["slot"].shape[0])
        _ent["slot"][:_nsl] = _flat[:_nsl].long()
        _s0 = _flat[:1].long()
        _pg = (_s0 // _rpp).clamp(max=swa_kv_cache.shape[0] - 1)
        _rw = _s0 % _rpp
        _ent["krow"][:1] = swa_kv_cache[_pg, _rw].reshape(1, -1)
        if _flat.numel() >= 2:
            # PAIR interpretation (block, offset) for BF16 plans — 1-D
            # 1-element index tensors ONLY (see docstring)
            _pair = _flat[:2].long().view(2, 1)
            _pb2 = _pair[0].clamp(max=swa_kv_cache.shape[0] - 1)
            _ow2 = _pair[1].clamp(min=0, max=_rpp - 1)
            _ent["krow"][1:2] = swa_kv_cache[_pb2, _ow2].reshape(1, -1)
    _o = attn_kwargs.get("ori_block_table")
    if _o is not None and _o.dim() == 2 and _o.shape[1] >= 4:
        _ent["obt"][:] = _o[0, :4].long()
        # ori trailing-window rows: gathered device-side so the REPLAYED
        # graph collects each step's own rows (seqused + table are fresh)
        _vis = (_sq.reshape(-1)[0].long().clamp(max=128)
                if _sq is not None and _sq.numel() >= 1
                else torch.zeros(1, dtype=torch.long, device=out.device))
        _cols = _o[0].long().clamp(0, swa_kv_cache.shape[0] - 1)
        _flat2 = swa_kv_cache.index_select(0, _cols).reshape(
            -1, swa_kv_cache.shape[-1])
        _idx = (torch.arange(128, device=out.device)
                - 128 + _vis.reshape(1)).clamp(
            min=0, max=_flat2.shape[0] - 1)
        _ent["ori_rows"][:] = _flat2.index_select(0, _idx)
    if _sq is not None and _sq.numel() >= 4:
        _ent["sq4"][:] = _sq.reshape(-1)[:4].long()
    _sa = attn_kwargs.get("metadata")
    if _sa is not None and _sa.numel() >= 1:
        _sfull = _sa.reshape(-1)
        _n = min(_sfull.numel(), 64)
        _ent["sas"][:_n] = _sfull[:_n].to(torch.int32)
    _se = attn_kwargs.get("cmp_sparse_indices")
    if _se is not None and _se.reshape(-1).numel() >= 8:
        _ent["sel"][:] = _se.reshape(-1)[:8].to(torch.int32)
    _snk2 = attn_kwargs.get("sinks")
    if _snk2 is not None and _snk2.numel() >= 1 \
            and _ent["sinks"].numel() >= 1:
        _n2 = min(int(_snk2.numel()), int(_ent["sinks"].numel()))
        _ent["sinks"].reshape(-1)[:_n2] = \
            _snk2.reshape(-1)[:_n2].to(out.dtype)
    return _key, _ent



def _v41_r57_dump(tag: str, tensor) -> None:
    """R57 first-divergence probe: one-shot per-tag dump, gated by the
    model-level V41_DUMP active flag (first real forward). L0 chain only."""
    try:
        from vllm_ascend.models.deepseek_v41.model import _V41_DUMP as _D
        if not _D.get("active") or _D.get("r57_" + tag):
            return
        _D["r57_" + tag] = True
        import os as _os

        import torch as _t
        t = tensor.detach()
        _t.save({"tensor": t[:64].cpu() if t.shape[0] > 64 else t.cpu(),
                 "shape": tuple(t.shape), "dtype": str(t.dtype)},
                _os.path.join(
                    _D["dir"],
                    f"r57_{tag}_r{_os.environ.get('VLLM_DP_RANK', '0')}"
                    f"_{_os.getpid()}.pt"))
    except Exception:  # noqa: BLE001 - probe must never kill serving
        pass


def v41_make_persist(num_reqs: int, num_blocks: int, device) -> dict:
    """Persistent per-step buffers the captured graph must read fresh."""
    persist = {
        "seq": torch.zeros(num_reqs, dtype=torch.int64, device=device),
        "bt0": torch.zeros(num_reqs, dtype=torch.int64, device=device),
        "num_reqs": torch.zeros(1, dtype=torch.int64, device=device),
        "seqused": torch.zeros(num_reqs, dtype=torch.int32, device=device),
        "ori_bt": torch.zeros(num_reqs, num_blocks, dtype=torch.int32, device=device),
        "cmp_bt": torch.zeros(num_reqs, num_blocks, dtype=torch.int32, device=device),
        "req_slot": torch.zeros(num_reqs, dtype=torch.int64, device=device),
        "smladbg": {},   # impl-registered per-layer graph-read debug tensors
        "stepdbg": {},
    }
    _V41_PERSIST_REGISTRY.append(persist)
    return persist


def v41_refresh_persist(persist: dict, seq_lens, block_table, num_reqs: int) -> None:
    """Rewrite the per-step values BEFORE each replay (eager build)."""
    n = min(int(num_reqs), persist["seq"].shape[0])
    nc = min(block_table.shape[1], persist["ori_bt"].shape[1])
    persist["seq"][:n] = seq_lens[:n].long()
    persist["bt0"][:n] = block_table[:n, 0].long()
    persist["num_reqs"].fill_(int(num_reqs))
    persist["seqused"][:n] = seq_lens[:n].to(torch.int32)
    persist["ori_bt"][:n, :nc] = block_table[:n, :nc].to(torch.int32)
    persist["cmp_bt"][:n, :nc] = block_table[:n, :nc].to(torch.int32)


def v41_write_req_slots(idx_mapping: torch.Tensor) -> None:
    """Runner hook (eager prepare, every step): publish the engine request
    slot id of each batch row into every registered persist set."""
    for p in _V41_PERSIST_REGISTRY:
        n = min(int(idx_mapping.shape[0]), p["req_slot"].shape[0])
        p["req_slot"][:n] = idx_mapping[:n].to(torch.int64)


def v41_io_drain_all() -> int:
    """Post-replay layer-IO drain (aclgraph runner calls this right after
    each FULL replay on this rank). The entries are single-slot buffers the
    NEXT replay rewrites — including the dummy batches every idle tick — so
    the values are guaranteed to belong to THIS replay only immediately
    after it; a later build-side drain would read whatever replay ran last
    (usually a dummy, seq==1)."""
    io_dir = _v41_io_drain_dir()
    if io_dir is None:
        return 0
    import torch as _t

    n = 0
    for _p in _V41_PERSIST_REGISTRY:
        for _lname, _ent in list(_p["smladbg"].items()):
            if not _lname.endswith("io"):
                continue
            try:
                if _v41_io_ent_is_dummy(_ent):
                    # idle-tick dummy content: never write (per-request
                    # replay evidence must be free of dummy ticks)
                    continue
                _v41_io_write_file(io_dir, _lname, _ent)
                n += 1
            except Exception:  # noqa: BLE001
                continue
    return n


@dataclass
class AscendDSAMetadata:
    """Metadata for MLACommon.
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    num_actual_tokens: int  # Number of tokens excluding padding.
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    # The dimension of the attention heads
    head_dim: int | None = None
    # chunked prefill by default if no attn_states passed
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    req_metadata: AscendDSAReqMetadata | None = None
    reshape_cache_event: torch.npu.Event = None

    # metadata for dsv4 indexer

    hadamard: torch.Tensor | None = None


DSAMetadataDict: TypeAlias = dict[str, AscendDSAMetadata]


@dataclass(frozen=True)
class AscendDSALayerMetadata:
    attention: AscendDSAMetadata | None
    swa: AscendDSAMetadata
    compressor: AscendCompressorMetadata | None = None
    indexer: AscendIndexerMetadata | None = None


def _require_req_metadata(metadata: AscendDSAMetadata) -> AscendDSAReqMetadata:
    assert metadata.req_metadata is not None
    return metadata.req_metadata


def get_dspark_sparse_sas_window(vllm_config: Any) -> tuple[int, int]:
    hf_config = vllm_config.model_config.hf_config
    window_size = int(hf_config.sliding_window)
    block_size = vllm_config.speculative_config.num_speculative_tokens
    return window_size + block_size - 1, 0


def _aligned_dspark_index_width(window_size: int, block_size: int, alignment: int = 128) -> int:
    min_width = int(window_size) + int(block_size)
    return ((min_width + alignment - 1) // alignment) * alignment


def build_dspark_swa_indices(
    block_table: torch.Tensor,
    num_speculative_tokens: int,
    window_size: int,
    block_size: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_decode_tokens: int | None = None,
    index_width: int | None = None,
    indices_output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build DSpark non-causal visible slot ids for a paged SWA cache.

    Each token in a draft block sees the trailing context window plus the
    whole current draft block. Invalid/paddedrows get lens=0 and -1 slots.
    """
    if index_width is None:
        index_width = _aligned_dspark_index_width(window_size, num_speculative_tokens)
    min_width = int(window_size) + int(num_speculative_tokens)
    if index_width < min_width:
        raise ValueError(
            "DSpark SWA index_width must cover window_size + block_size: "
            f"index_width={index_width}, required={min_width}"
        )
    if query_start_loc is None or seq_lens is None:
        raise ValueError("DSpark SWA query_start_loc and seq_lens must both be provided")

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    prefix_lens = seq_lens - query_lens
    start_pos = (prefix_lens - int(window_size)).clamp(min=0)
    visible_lens = seq_lens - start_pos

    # Per-request visible-position grid [req_count, index_width]. Columns
    # j >= visible_len are out of range and masked to -1 below.
    cols = torch.arange(index_width, device=start_pos.device)
    col_mask = cols[None, :] < visible_lens[:, None]
    pos = start_pos[:, None] + cols[None, :]
    block_nums = pos // block_size
    # Clamp to valid block-table columns so gather never goes OOB on the
    # out-of-range columns (their results are discarded by col_mask anyway).
    safe_nums = block_nums.clamp(min=0, max=int(block_table.shape[1]) - 1)
    block_offsets = pos % block_size
    block_ids = torch.gather(block_table, 1, safe_nums)
    slot_ids = (block_ids * block_size + block_offsets).to(torch.int32)
    slot_ids = slot_ids.where(col_mask, torch.full_like(slot_ids, -1))

    per_token_slots = torch.repeat_interleave(slot_ids, query_lens, dim=0, output_size=num_decode_tokens).unsqueeze(1)
    per_token_lens = torch.repeat_interleave(visible_lens, query_lens, dim=0, output_size=num_decode_tokens)

    if indices_output is not None:
        if indices_output.shape != per_token_slots.shape:
            raise ValueError(
                "DSpark SWA indices output shape does not match active metadata: "
                f"output={tuple(indices_output.shape)}, active={tuple(per_token_slots.shape)}"
            )
        indices_output.copy_(per_token_slots)
        per_token_slots = indices_output

    return per_token_slots, per_token_lens


class AscendDSAMetadataBuilder(AttentionMetadataBuilder[AscendDSAMetadata]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        kv_cache_spec: AscendMLAAttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendDSAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        self.kv_cache_spec = kv_cache_spec
        self.metadata_cls = metadata_cls if metadata_cls is not None else AscendDSAMetadata
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        self.logical_block_size = kv_cache_spec.block_size
        self.storage_block_size = kv_cache_spec.storage_block_size
        scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        self.spec_slot_mapping = None
        if get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION) and not is_a5_bf16_kv_enabled(
            vllm_config
        ):
            self.slot_mapping_shape = (vllm_config.scheduler_config.max_num_batched_tokens,)  # type: ignore
        else:
            self.slot_mapping_shape = (vllm_config.scheduler_config.max_num_batched_tokens, 2)  # type: ignore
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.spec_slot_mapping = [
                torch.zeros(self.slot_mapping_shape, dtype=torch.int32, device=self.device)
                for _ in range(spec_token_num)
            ]
            self.spec_sas_metadata = [
                torch.zeros(
                    DSA_METADATA_BUFFER_SIZE,
                    dtype=torch.int32,
                    device=self.device,
                )
                for _ in range(spec_token_num)
            ]
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )

        self.reorder_batch_threshold = self.decode_threshold
        self.num_decodes = 0
        self.num_prefills = 0
        self.num_decode_tokens = 0
        self.num_prefill_tokens = 0
        self.num_actual_tokens: int | None = None
        self.block_table: torch.Tensor = None
        self.common_ratio_to_sas_metadata: dict | None = None
        self.seq_lens: torch.Tensor = None

        self.compressor_ratio = getattr(kv_cache_spec, "compress_ratio", 0)
        # DeepSeek V4.1: a compressed layer whose compress_ratio is 1 must be
        # distinguished from a pure sliding-window layer. The compressed layer
        # owns an AscendMLAAttentionSpec (compress_ratio=1), the pure SWA layer
        # an AscendSlidingWindowMLASpec, so key off the spec type.
        self.is_compressed_layer = isinstance(kv_cache_spec, AscendMLAAttentionSpec)
        self.hadamard = None
        self._init_hadamard(layer_names)
        self.start_pos_prefill: torch.Tensor = torch.zeros(
            scheduler_config.max_num_seqs, dtype=torch.int32, device=self.device
        )
        self.sas_metadata_buffer: torch.Tensor = torch.zeros(
            DSA_METADATA_BUFFER_SIZE, dtype=torch.int32, device=self.device
        )
        self.qli_metadata_buffer: torch.Tensor = torch.zeros(
            DSA_METADATA_BUFFER_SIZE, dtype=torch.int32, device=self.device
        )
        self._device_metadata_enabled = False
        self._device_metadata_tasks: tuple[DeviceMetadataTask, ...] = ()
        self.cu_seqlens_ori_kv = torch.tensor([], device=self.device)
        self.cu_seqlens_cmp_kv = torch.tensor([], device=self.device)
        self.seqused_q = torch.tensor([], device=self.device)
        self._zero_i32 = torch.tensor([0], device=self.device, dtype=torch.int32)
        # Note(qcs): we use two dimension slot_mapping for kvcache with shape
        # [block_nums, block_size, head_num, head_dim]
        self.slot_mapping = torch.zeros(self.slot_mapping_shape, dtype=torch.int32, device=self.device)
        self.compressor_metadata_buffers: CompressorMetadataOutput | None = None
        self.dspark_swa_indices_buffer: torch.Tensor | None = None

    def _init_hadamard(self, layer_names: list[str]) -> None:
        hf_config = self.model_config.hf_config
        if hf_config.model_type != "deepseek_v4":
            return

        indexer_head_dim = hf_config.index_head_dim
        try:
            from scipy.linalg import hadamard  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError("Please install scipy") from e
        log_dim = math.ceil(math.log2(indexer_head_dim))
        dim_padded = 2**log_dim
        self.hadamard = get_or_register_attention_buffer(
            self.vllm_config,
            layer_names,
            "_dsa_hadamard",
            lambda: torch.tensor(hadamard(dim_padded, dtype=float), dtype=torch.float, device=self.device).to(
                torch.bfloat16
            ),
        )

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendDSAMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.UNIFORM_BATCH

    def reorder_batch(self, input_batch: "NPUInputBatch", scheduler_output: "SchedulerOutput") -> bool:
        # We now want to reorder the batch so that the "decode" requests are at
        # the front and the "prefill" requests are at the using the least amount
        # swaps possible. (NOTE for now we loosely use "decode" to mean requests
        # where attention is likely memory-bound and "prefill" to mean requests
        # where attention is likely compute-bound, TODO(lucas): figure out a
        # better naming here)
        decodes = []
        prefills = []

        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            if num_tokens <= self.decode_threshold:
                decodes.append(i)
            else:
                prefills.append(i)

        # We hope that this is fairly minimal since decodes
        # should be around for a number of iterations so hopefully they are
        # relatively stationary (and new request are generally appended to the
        # persistent batch so already should be at the back)
        # To achieve this we loop over the decodes in descending order and
        # the prefills in ascending order. We swap decodes from the  "back"
        # i.e. past where the last decode should be in the reodorered with
        # prefills from the front of the batch.
        # `decodes` and `prefills` are already in ascending order just based on
        # the above loop
        num_decodes = len(decodes)
        num_prefills = len(prefills)
        first_prefill = 0
        modified_batch = False

        for i in range(1, min(num_decodes, num_prefills) + 1):
            # If the decode is at the "back" of the batch, i, we can swap it
            # with the prefill closest to the front of the batch
            if decodes[num_decodes - i] >= num_decodes:
                input_batch.swap_states(prefills[first_prefill], decodes[num_decodes - i])
                first_prefill += 1
                modified_batch = True
            else:
                break

        # Save for next `build` call
        # TODO(lucas): this is a bit of a hack, we should probably have a
        # better way of doing this
        return modified_batch

    def set_num_actual_tokens(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ):
        self.num_actual_tokens = common_attn_metadata.num_actual_tokens

    def _num_compressor_metadata_rows(self, num_reqs: int) -> int:
        assert self.num_actual_tokens is not None
        num_tokens = self.num_actual_tokens
        return min(num_tokens, num_tokens // self.compressor_ratio + num_reqs)

    def build_for_cudagraph_capture(self, common_attn_metadata: AscendCommonAttentionMetadata) -> AscendDSAMetadata:
        """Delegate to build() because DSA needs shared request metadata."""
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            num_actual_reqs=common_attn_metadata.num_reqs,
            common_ratio_to_sas_metadata={},
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendDSAMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_reqs = kwargs.get("num_actual_reqs")
        self.common_ratio_to_sas_metadata = kwargs.get("common_ratio_to_sas_metadata")
        assert self.common_ratio_to_sas_metadata is not None
        self.set_num_actual_tokens(common_attn_metadata)
        num_input_tokens = common_attn_metadata.num_input_tokens

        if self.common_ratio_to_sas_metadata.get("num_decodes") is None:
            self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = (
                split_decodes_and_prefills(
                    common_attn_metadata,
                    decode_threshold=self.decode_threshold,
                )
            )
            self.common_ratio_to_sas_metadata["num_decodes"] = self.num_decodes
            self.common_ratio_to_sas_metadata["num_prefills"] = self.num_prefills
            self.common_ratio_to_sas_metadata["num_decode_tokens"] = self.num_decode_tokens
            self.common_ratio_to_sas_metadata["num_prefill_tokens"] = self.num_prefill_tokens
            assert self.num_decodes + self.num_prefills == num_reqs
            assert self.num_decode_tokens + self.num_prefill_tokens == common_attn_metadata.num_actual_tokens
            self.seq_lens = common_attn_metadata.seq_lens[:num_reqs]
            self.common_ratio_to_sas_metadata["seq_lens"] = self.seq_lens
            # Prefer _seq_lens_cpu (always available, updated during draft
            # iterations) over seq_lens_cpu (None in async spec decode mode).
            if common_attn_metadata._seq_lens_cpu is not None:
                seq_lens_cpu = common_attn_metadata._seq_lens_cpu
            elif common_attn_metadata.seq_lens_cpu is not None:
                seq_lens_cpu = common_attn_metadata.seq_lens_cpu
            else:
                seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
            self.common_ratio_to_sas_metadata["seq_lens_cpu"] = seq_lens_cpu
            input_positions = common_attn_metadata.positions[:num_input_tokens].long()
            cos, sin = get_cos_and_sin_dsa(
                input_positions,
                use_cache=self.num_prefills == 0,
            )
            self.common_ratio_to_sas_metadata["cos"] = cos
            self.common_ratio_to_sas_metadata["sin"] = sin
            if __import__("os").environ.get("V41_STEP_LOG", "0") == "1":
                # R19 per-step composition (one line per engine per step, only
                # in the first builder that computes the shared metadata):
                # exact request mix at a crash + live proof of the
                # num_actual_reqs flow the refstore demand relies on.
                import os as _os_sl

                print(f"[V41_STEP] rank={_os_sl.environ.get('VLLM_DP_RANK', '?')} "
                      f"pid={_os_sl.getpid()} reqs={num_reqs} "
                      f"actual={num_actual_reqs} tokens={num_input_tokens} "
                      f"actual_tokens={self.num_actual_tokens} "
                      f"prefills={self.num_prefills} decodes={self.num_decodes}",
                      flush=True)
        else:
            self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = (
                self.common_ratio_to_sas_metadata["num_decodes"],
                self.common_ratio_to_sas_metadata["num_prefills"],
                self.common_ratio_to_sas_metadata["num_decode_tokens"],
                self.common_ratio_to_sas_metadata["num_prefill_tokens"],
            )
            self.seq_lens = self.common_ratio_to_sas_metadata["seq_lens"]
            seq_lens_cpu = self.common_ratio_to_sas_metadata["seq_lens_cpu"]
            cos = self.common_ratio_to_sas_metadata["cos"]
            sin = self.common_ratio_to_sas_metadata["sin"]

        # CommonAttentionMetadata uses logical raw-token slots. They directly
        # describe only uncompressed SWA/state caches; C4/C128 physical slots
        # are generated later from the logical block table by compressor_metadata.
        if self.compressor_ratio <= 1:
            slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
            self.slot_mapping[:num_input_tokens] = get_dsa_attn_kv_plan(self.vllm_config).format_dsa_slot_mapping(
                slot_mapping, self.storage_block_size
            )

        self.block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        # Graph-decode freshness: the captured V4.1 device path must read
        # per-step values from PERSISTENT buffers (captured per-step tensors
        # like common_attn_metadata.seq_lens are stale under replay — the
        # runner never writes them again). The persist set is handed to the
        # impls INSIDE AscendDSAReqMetadata (same mechanism as
        # sas_metadata_buffer): at capture the kernels bind these storages,
        # and this refresh (every eager build, before each replay) rewrites
        # the contents. Builder-local attributes NEVER reach the impl — the
        # objects are different instances.
        if __import__("os").environ.get("V41_DEVICE_INDEXER", "0") == "1":
            if getattr(self, "_v41_persist", None) is None:
                self._v41_persist = v41_make_persist(
                    int(self.vllm_config.scheduler_config.max_num_seqs),
                    int(self.block_table.shape[1]),
                    self.device,
                )
            v41_refresh_persist(self._v41_persist, self.seq_lens, self.block_table, num_reqs)
        req_metadata = self.build_req_metadata(
            common_attn_metadata=common_attn_metadata,
            seq_lens_cpu=seq_lens_cpu,
            num_actual_reqs=num_actual_reqs,
            cos=cos,
            sin=sin,
            full_graph_mode=kwargs.get("full_graph_mode", False),
        )
        if __import__("os").environ.get("V41_GRAPH_STEPDBG", "0") == "1":
            import json as _json

            _p = getattr(self, "_v41_persist", None)
            rec = {"phase": "build",
                   "worker": __import__("os").environ.get("RANK", "0"),
                   "group": f"c{self.compressor_ratio}",
                   "num_reqs": int(num_reqs),
                   "seq_head": self.seq_lens[:4].detach().cpu().tolist(),
                   "bt_head": self.block_table[:2, :4].detach().cpu().tolist()}
            if _p is not None:
                rec["graph_seq"] = _p["seq"][:4].detach().cpu().tolist()
                rec["req_slot"] = _p["req_slot"][:4].detach().cpu().tolist()
                # per-layer graph-read samples: written by the CAPTURED kernels
                # (impl registers them into the persist set at capture); the
                # values below are whatever the last completed execution left
                # — align via the in-graph step counters.
                for _lname, _sd in _p["smladbg"].items():
                    rec.setdefault("smla", []).append({
                        "layer": _lname,
                        "graph_step": int(_sd["step"].detach().cpu().item()),
                        "seqused": _sd["seqused"].detach().cpu().tolist(),
                        "ori_bt": _sd["ori_bt"].detach().cpu().tolist(),
                        "cmp_bt": _sd["cmp_bt"].detach().cpu().tolist(),
                        "cmp_idx": _sd["cmp_idx"].detach().cpu().tolist(),
                        "sas": _sd["sas"].detach().cpu().tolist(),
                    })
                for _lname, _sd in _p["stepdbg"].items():
                    rec.setdefault("stepdbg", []).append({
                        "layer": _lname,
                        "graph_step": int(_sd["step"].detach().cpu().item()),
                        "akl": _sd["akl"].detach().cpu().tolist(),
                        "seq": _sd["seq"].detach().cpu().tolist(),
                        "bt": _sd["bt"].detach().cpu().tolist(),
                        "topk": _sd["topk"].detach().cpu().tolist(),
                    })
            try:
                _sb = self.sas_metadata_buffer
                rec["sas_buf_head"] = _sb[:16].detach().cpu().tolist()
                rec["sas_sum"] = int(_sb.sum().detach().cpu().item())
                rec["sas_nz"] = int((_sb != 0).sum().detach().cpu().item())
            except Exception:  # noqa: BLE001 - debug only
                rec["sas_buf_head"] = None
            try:
                import os as _os

                _os.makedirs("/ws/evidence/p85run1", exist_ok=True)
                with open("/ws/evidence/p85run1/smladbg.jsonl", "a") as f:
                    f.write(_json.dumps(rec) + "\n")
            except Exception:  # noqa: BLE001 - debug only
                pass

        # layer-IO drain (independent of V41_GRAPH_STEPDBG): files for a
        # captured step are written here — this runs on the eager Python of
        # the NEXT step's build, i.e. strictly AFTER the replay executed
        # (the entry's in-graph step counter must have advanced).
        _io_dir = _v41_layer_io_dir()
        if _io_dir is not None:
            _p_io = getattr(self, "_v41_persist", None)
            if _p_io is not None:
                _last_io = getattr(self, "_v41_io_last", None)
                if _last_io is None:
                    _last_io = self._v41_io_last = {}
                for _lname, _ent in list(_p_io["smladbg"].items()):
                    if not _lname.endswith("#io"):
                        continue
                    _st = int(_ent["step"].detach().cpu().item())
                    if _st <= _last_io.get(_lname, 0):
                        continue
                    _last_io[_lname] = _st
                    _v41_io_write_file(_io_dir, _lname, _ent)


        return self.metadata_cls(  # type: ignore
            num_actual_tokens=self.num_actual_tokens,
            head_dim=self.model_config.get_head_size(),
            num_decodes=self.num_decodes,
            num_decode_tokens=self.num_decode_tokens,
            num_prefills=self.num_prefills,
            attn_state=common_attn_metadata.attn_state,
            req_metadata=req_metadata,
            hadamard=self.hadamard,
        )

    def _build_sas_metadata(
        self,
        metadata_cache: dict,
        layer_name: str,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seqlen_q: int | torch.Tensor,
        max_seqlen_kv: int | torch.Tensor,
        cu_seqlens_ori_kv: torch.Tensor | None,
        cu_seqlens_cmp_kv: torch.Tensor | None,
    ) -> torch.Tensor:
        sas_metadata = metadata_cache.get(layer_name)
        if sas_metadata is None:
            tp_size = get_tensor_model_parallel_world_size()
            n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size
            index_topk = self.model_config.hf_config.index_topk
            if self.is_compressed_layer:
                # Compressed layer (V4 ratio 4/128, V4.1 ratio 1/2): the
                # compressed stream exists and its ratio is the spec's value.
                cmp_ratio = self.compressor_ratio
            else:
                cmp_ratio = _dsa_swa_only_cmp_ratio(self.compressor_ratio, self.vllm_config)
            kv_plan = get_dsa_attn_kv_plan(self.vllm_config)
            metadata_op = kv_plan.get_dsa_sparse_attn_metadata_op()
            metadata_kwargs = kv_plan.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)
            sas_metadata = metadata_op(
                **metadata_kwargs,
                num_heads_q=n_local_heads,
                num_heads_kv=1,
                head_dim=self.model_config.get_head_size(),
                cu_seqlens_q=query_start_loc,
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                seqused_q=self.seqused_q,
                seqused_kv=seq_lens,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=len(seq_lens),
                cmp_topk=index_topk
                if _dsa_uses_indexer(self.compressor_ratio, self.is_compressed_layer)
                else 0,
                cmp_ratio=cmp_ratio,
                ori_mask_mode=4,  # 4:sliding window
                cmp_mask_mode=3,  # 3:causal
                ori_win_left=self.model_config.hf_config.sliding_window - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv=_dsa_layout_kv(self.vllm_config),
                has_ori_kv=True,
                has_cmp_kv=self.is_compressed_layer,
            )
            metadata_cache[layer_name] = sas_metadata

        self.sas_metadata_buffer[:DSA_METADATA_BUFFER_SIZE] = sas_metadata
        return self.sas_metadata_buffer

    def _build_qli_metadata(
        self,
        metadata_cache: dict,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_kv: int,
    ) -> torch.Tensor:
        qli_metadata = metadata_cache.get("qli")
        if qli_metadata is None:
            qli_metadata = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer_metadata(
                actual_seq_lengths_query=query_start_loc[1:].clone(),
                actual_seq_lengths_key=seq_lens.clone(),
                num_heads_q=self.model_config.hf_config.index_n_heads,  # 64
                num_heads_k=1,
                head_dim=self.model_config.hf_config.index_head_dim,  # 128
                query_quant_mode=0,
                key_quant_mode=0,
                batch_size=len(seq_lens),
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_kv,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.model_config.hf_config.index_topk,  # 512
                sparse_mode=3,
                pre_tokens=(1 << 63) - 1,
                next_tokens=(1 << 63) - 1,
                cmp_ratio=self.compressor_ratio if self.is_compressed_layer else 4,
                device=str(self.seqused_q.device),
            )
            metadata_cache["qli"] = qli_metadata

        self.qli_metadata_buffer[:DSA_METADATA_BUFFER_SIZE] = qli_metadata
        return self.qli_metadata_buffer

    def enable_device_metadata(self) -> None:
        self._device_metadata_enabled = True
        if self.is_compressed_layer and self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL:
            max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
            output_shape = (max_tokens, 1, 1, self.model_config.hf_config.qk_rope_head_dim)
            self.compressor_metadata_buffers = (
                torch.empty(output_shape, dtype=torch.float32, device=self.device),
                torch.empty(output_shape, dtype=torch.float32, device=self.device),
                self.slot_mapping,
            )

    def enable_dspark_device_metadata(self, max_num_tokens: int) -> None:
        self.enable_device_metadata()
        assert self.speculative_config is not None
        index_width = _aligned_dspark_index_width(
            self.model_config.hf_config.sliding_window,
            self.speculative_config.num_speculative_tokens,
        )
        self.dspark_swa_indices_buffer = torch.empty(
            (max_num_tokens, 1, index_width),
            dtype=torch.int32,
            device=self.device,
        )

    def take_device_metadata_tasks(self) -> tuple[DeviceMetadataTask, ...]:
        tasks = self._device_metadata_tasks
        self._device_metadata_tasks = ()
        return tasks

    def build_req_metadata(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        seq_lens_cpu: torch.Tensor,
        num_actual_reqs: int | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        full_graph_mode: bool = False,
    ) -> AscendDSAReqMetadata:
        assert self.common_ratio_to_sas_metadata is not None
        metadata_cache = self.common_ratio_to_sas_metadata
        assert self.num_actual_tokens is not None
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc[: num_reqs + 1]
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
        seq_lens = self.seq_lens[:num_reqs]
        seq_lens_q = query_start_loc[1:] - query_start_loc[:-1]
        max_seqlen_q = torch.max(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).item()
        max_seqlen_kv = torch.max(seq_lens_cpu[:num_reqs]).item()
        has_prefill = self.num_prefills > 0

        self.start_pos_prefill.fill_(0)
        self.start_pos_prefill[:num_reqs] = seq_lens - seq_lens_q
        if num_actual_reqs is None:
            num_actual_reqs = num_reqs
        else:
            num_actual_reqs = min(num_actual_reqs, num_reqs)
            if num_actual_reqs < num_reqs:
                self.start_pos_prefill[num_actual_reqs:num_reqs].fill_(0)
                self.block_table[num_actual_reqs:num_reqs, ...].fill_(0)
        layer_name = f"c{self.compressor_ratio}"
        cu_seqlens_ori_kv = None
        cu_seqlens_cmp_kv = None
        dspark_swa_indices = None
        ori_win_left, ori_win_right = self.model_config.hf_config.sliding_window - 1, 0
        if not has_prefill and not common_attn_metadata.causal:
            # DSpark non-causal parallel drafting: every draft query attends to
            # the trailing context window plus the whole current draft block.
            # Not gated on the SAS metadata cache: the indices depend on the
            # current step's block table / sequence lengths, so they must be
            # rebuilt whenever a DSpark draft step runs.
            assert self.speculative_config is not None
            dspark_swa_indices, _ = build_dspark_swa_indices(
                self.block_table[: self.num_decodes],
                self.speculative_config.num_speculative_tokens,
                self.model_config.hf_config.sliding_window,
                self.storage_block_size,
                query_start_loc[: self.num_decodes + 1],
                self.seq_lens[: self.num_decodes],
                self.num_decode_tokens,
            )
            dspark_swa_indices = dspark_swa_indices[: self.num_decode_tokens]
            ori_win_left, ori_win_right = get_dspark_sparse_sas_window(self.vllm_config)
        if not has_prefill and self.common_ratio_to_sas_metadata.get(layer_name) is None:
            cu_seqlens_ori_kv = DeviceOperator.get_dsa_decode_cu_seqlens_ori_kv(
                self.common_ratio_to_sas_metadata,
                "cu_seqlens_ori_kv",
                seq_lens,
                num_reqs,
                self._zero_i32,
                self.cu_seqlens_ori_kv,
            )
            cu_seqlens_cmp_kv = DeviceOperator.get_dsa_decode_cu_seqlens_cmp_kv(self.cu_seqlens_cmp_kv)
        elif has_prefill:
            cu_seqlens_ori_kv = query_start_loc

        if self._device_metadata_enabled:

            def build_sas_metadata() -> None:
                self._build_sas_metadata(
                    metadata_cache=metadata_cache,
                    layer_name=layer_name,
                    query_start_loc=query_start_loc,
                    seq_lens=seq_lens,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                    cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                )

            def build_qli_metadata() -> None:
                self._build_qli_metadata(
                    metadata_cache=metadata_cache,
                    query_start_loc=query_start_loc,
                    seq_lens=seq_lens,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                )

            # The fused lightning-indexer QLI metadata only serves the V4
            # ratio-4 path; V4.1 ratio 1/2 groups select top-k with the torch
            # reference indexer and must not build (or run) the aicpu op.
            _needs_qli = (
                self.compressor_ratio == 4
                and _dsa_uses_indexer(self.compressor_ratio, self.is_compressed_layer)
            )
            if _needs_qli:
                self._device_metadata_tasks = (
                    DeviceMetadataTask(DeviceMetadataStage.INDEXER, build_qli_metadata, id(self.qli_metadata_buffer)),
                    DeviceMetadataTask(DeviceMetadataStage.ATTENTION, build_sas_metadata, id(self.sas_metadata_buffer)),
                )
            else:
                self._device_metadata_tasks = (
                    DeviceMetadataTask(DeviceMetadataStage.ATTENTION, build_sas_metadata, id(self.sas_metadata_buffer)),
                )
            sas_metadata = self.sas_metadata_buffer
            qli_metadata = self.qli_metadata_buffer if _needs_qli else None
        else:
            self._device_metadata_tasks = ()
            sas_metadata = self._build_sas_metadata(
                metadata_cache=metadata_cache,
                layer_name=layer_name,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            )
            # V4.1 ratio 1/2 groups run the torch reference indexer: no QLI.
            qli_metadata = (
                self._build_qli_metadata(
                    metadata_cache=metadata_cache,
                    query_start_loc=query_start_loc,
                    seq_lens=seq_lens,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                )
                if self.compressor_ratio == 4
                and _dsa_uses_indexer(self.compressor_ratio, self.is_compressed_layer)
                else None
            )

        full_compress_cos, full_compress_sin = None, None
        if self.is_compressed_layer:
            if full_graph_mode:
                num_tokens = common_attn_metadata.num_input_tokens
                num_compressed_tokens = min(num_tokens, num_tokens // self.compressor_ratio + num_reqs)
                num_actual_reqs = num_reqs
            else:
                num_compressed_tokens = self._num_compressor_metadata_rows(num_reqs)
            full_compress_cos, full_compress_sin = get_full_cos_and_sin_dsa(layer_name)
            slot_mapping = None
        else:
            num_compressed_tokens = self.num_actual_tokens
            slot_mapping = self.slot_mapping[: self.num_actual_tokens]

        req_metadata = AscendDSAReqMetadata(
            block_table=self.block_table[:num_reqs, ...],
            seq_lens=seq_lens,
            slot_mapping=slot_mapping,
            storage_block_size=self.storage_block_size,
            query_start_loc=query_start_loc,
            num_compressed_tokens=num_compressed_tokens,
            sin=sin,
            cos=cos,
            full_compress_sin=full_compress_sin,
            full_compress_cos=full_compress_cos,
            start_pos=self.start_pos_prefill[:num_reqs],
            num_actual_reqs=num_actual_reqs,
            sas_metadata=sas_metadata,
            qli_metadata=qli_metadata,
            attn_mask=None,
            cu_cmp_seqlen_list=cu_seqlens_cmp_kv,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            dspark_swa_indices=dspark_swa_indices,
            v41_persist=getattr(self, "_v41_persist", None),
        )
        if self._device_metadata_enabled and self.compressor_metadata_buffers is not None:
            assert num_compressed_tokens is not None
            buffers = self.compressor_metadata_buffers
            outputs = (
                buffers[0][:num_compressed_tokens],
                buffers[1][:num_compressed_tokens],
                buffers[2][:num_compressed_tokens],
            )
            group_id = id(buffers[0])
            req_metadata.compressor_metadata = outputs
            req_metadata.compressor_metadata_group_id = group_id
            self._device_metadata_tasks = (
                DeviceMetadataTask(
                    DeviceMetadataStage.COMPRESSOR,
                    lambda: build_compressor_metadata_out(
                        req_metadata, self.compressor_ratio, outputs, self.vllm_config
                    ),
                    group_id,
                ),
                *self._device_metadata_tasks,
            )
        if self._device_metadata_enabled and self._device_metadata_tasks:
            # V2 runner + DSA: AscendDSAImpl.update_graph_params is a no-op,
            # so NOTHING would ever run these tasks — sas_metadata_buffer
            # would keep the last inline-written content (a PREFILL-shaped
            # plan after the first prefill) while the replayed decode graph
            # runs a T=1 kernel: the kernel walks a 6-query plan over a
            # 1-query tensor -> OOB -> aicore trap (SparseFlashMla,
            # subErrType 0x4; eager mode refreshes inline and never traps).
            # Execute the tasks inline HERE so every replay reads THIS
            # step's plan. Re-running them later via the V1 task path is
            # idempotent (same inputs -> same buffer content).
            for _task in self._device_metadata_tasks:
                _task.run()
        return req_metadata

    def build_for_drafting(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        draft_index: int,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendDSAMetadata:
        assert self.compressor_ratio <= 1, "vLLM-Ascend only support SWA-layer for Deepseek-V4 now."
        self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.decode_threshold,
            )
        )
        num_reqs = common_attn_metadata.num_reqs
        num_input_tokens = common_attn_metadata.num_input_tokens
        self.num_actual_tokens = common_attn_metadata.num_actual_tokens
        self.seq_lens = common_attn_metadata.seq_lens[:num_reqs]
        self.block_table = common_attn_metadata.block_table_tensor[:num_reqs]

        input_positions = common_attn_metadata.positions[:num_input_tokens].long()
        if self.num_prefills:
            cos, sin = get_cos_and_sin_dsa(input_positions)
        else:
            cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=True, draft_index=draft_index)
        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        assert self.spec_slot_mapping is not None
        self.spec_slot_mapping[draft_index - 1][:num_input_tokens] = get_dsa_attn_kv_plan(
            self.vllm_config
        ).format_dsa_slot_mapping(slot_mapping, self.storage_block_size)
        req_metadata = self.build_req_metadata_for_drafting(
            draft_index=draft_index,
            common_attn_metadata=common_attn_metadata,
            cos=cos,
            sin=sin,
        )

        return self.metadata_cls(  # type: ignore
            num_actual_tokens=self.num_actual_tokens,
            head_dim=self.model_config.get_head_size(),
            num_decodes=self.num_decodes,
            num_decode_tokens=self.num_decode_tokens,
            num_prefills=self.num_prefills,
            attn_state=common_attn_metadata.attn_state,
            req_metadata=req_metadata,
            hadamard=None,
        )

    def build_req_metadata_for_drafting(
        self,
        draft_index: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        cos,
        sin,
    ) -> AscendDSAReqMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc[: num_reqs + 1]
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]
        seq_lens = self.seq_lens[:num_reqs]
        seq_lens_q = query_start_loc[1:] - query_start_loc[:-1]
        max_seqlen_q = torch.max(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).item()
        if common_attn_metadata._seq_lens_cpu is not None:
            seq_lens_cpu = common_attn_metadata._seq_lens_cpu
        elif common_attn_metadata.seq_lens_cpu is not None:
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        else:
            seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
        max_seqlen_kv = torch.max(seq_lens_cpu[:num_reqs]).item()
        has_prefill = self.num_prefills > 0

        dspark_swa_indices = None
        build_dspark_swa = None
        ori_win_left = self.model_config.hf_config.sliding_window - 1
        ori_win_right = 0
        if not common_attn_metadata.causal:
            assert self.speculative_config is not None
            dspark_swa_args = (
                self.block_table[:num_reqs],
                self.speculative_config.num_speculative_tokens,
                self.model_config.hf_config.sliding_window,
                self.storage_block_size,
                query_start_loc,
                seq_lens,
                self.num_actual_tokens,
            )
            if self._device_metadata_enabled and not has_prefill:
                if self.dspark_swa_indices_buffer is None:
                    raise RuntimeError(
                        "DSpark device metadata buffers must be initialized before building draft attention metadata"
                    )
                if self.num_actual_tokens > self.dspark_swa_indices_buffer.shape[0]:
                    raise ValueError(
                        "DSpark SWA metadata rows exceed the persistent buffer capacity: "
                        f"active={self.num_actual_tokens}, capacity={self.dspark_swa_indices_buffer.shape[0]}"
                    )
                dspark_swa_indices = self.dspark_swa_indices_buffer[: self.num_actual_tokens]
                build_dspark_swa = lambda: build_dspark_swa_indices(
                    *dspark_swa_args,
                    indices_output=dspark_swa_indices,
                )
            else:
                dspark_swa_indices, _ = build_dspark_swa_indices(*dspark_swa_args)
                dspark_swa_indices = dspark_swa_indices[: self.num_actual_tokens]
            ori_win_left, ori_win_right = get_dspark_sparse_sas_window(self.vllm_config)

        cu_seqlens_ori_kv = (
            query_start_loc
            if has_prefill
            else DeviceOperator.get_dsa_decode_cu_seqlens_ori_kv(
                None,
                "draft_cu_seqlens_ori_kv",
                seq_lens,
                num_reqs,
                self._zero_i32,
                self.cu_seqlens_ori_kv,
            )
        )
        cu_seqlens_cmp_kv = (
            None if has_prefill else DeviceOperator.get_dsa_decode_cu_seqlens_cmp_kv(self.cu_seqlens_cmp_kv)
        )
        kv_plan = get_dsa_attn_kv_plan(self.vllm_config)
        metadata_op = kv_plan.get_dsa_sparse_attn_metadata_op()
        metadata_kwargs = kv_plan.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size

        def build_attention_metadata() -> torch.Tensor:
            if build_dspark_swa is not None:
                build_dspark_swa()
            result = metadata_op(
                **metadata_kwargs,
                num_heads_q=n_local_heads,
                num_heads_kv=1,
                head_dim=self.model_config.get_head_size(),
                cu_seqlens_q=query_start_loc,
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                seqused_q=self.seqused_q,
                seqused_kv=seq_lens,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=num_reqs,
                cmp_ratio=1,
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
                layout_q="TND",
                layout_kv=_dsa_layout_kv(self.vllm_config),
                has_ori_kv=True,
                has_cmp_kv=False,
            )
            if not has_prefill:
                assert self.spec_sas_metadata is not None
                self.spec_sas_metadata[draft_index - 1][:DSA_METADATA_BUFFER_SIZE].copy_(
                    result[:DSA_METADATA_BUFFER_SIZE]
                )
                return self.spec_sas_metadata[draft_index - 1]
            return result

        if build_dspark_swa is not None:
            sas_metadata = self.spec_sas_metadata[draft_index - 1]

            def run_attention_metadata() -> None:
                build_attention_metadata()

            self._device_metadata_tasks = (
                DeviceMetadataTask(DeviceMetadataStage.ATTENTION, run_attention_metadata, id(sas_metadata)),
            )
        else:
            self._device_metadata_tasks = ()
            sas_metadata = build_attention_metadata()

        assert self.spec_slot_mapping is not None
        slot_mapping = self.spec_slot_mapping[draft_index - 1][: self.num_actual_tokens]
        return AscendDSAReqMetadata(
            block_table=self.block_table[:num_reqs, ...],
            seq_lens=seq_lens,
            slot_mapping=slot_mapping,
            storage_block_size=self.storage_block_size,
            query_start_loc=query_start_loc,
            num_compressed_tokens=self.num_actual_tokens,
            sin=sin,
            cos=cos,
            start_pos=self.seq_lens[:num_reqs] - seq_lens_q,
            num_actual_reqs=num_reqs,
            sas_metadata=sas_metadata,
            qli_metadata=None,
            attn_mask=None,
            cu_cmp_seqlen_list=cu_seqlens_cmp_kv,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            dspark_swa_indices=dspark_swa_indices,
            v41_persist=getattr(self, "_v41_persist", None),
        )

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
        **kwargs,
    ):
        if attn_state in {AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding}:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly and SpecDecoding state"
            )

        assert attn_metadata is not None
        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendDSAImpl(AttentionImplBase[Any]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        n_heads: int,
        scale: float,
        n_local_heads: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int | None,
        nope_head_dim: int,
        n_groups: int,
        n_local_groups: int,
        window_size: int,
        compress_ratio: int,
        **kwargs,
    ):
        self.vllm_config = kwargs["vllm_config"]
        self.num_heads = n_heads
        self.n_local_heads = n_local_heads
        self.scale = scale
        self.o_lora_rank = o_lora_rank
        self.nope_head_dim = nope_head_dim
        self.rope_head_dim = rope_head_dim
        self.head_dim = head_dim
        self.n_group = n_groups
        self.n_local_groups = n_local_groups
        self.window_size = window_size
        self.q_lora_rank = q_lora_rank
        self.compress_ratio = compress_ratio
        # DeepSeek V4.1: compress_ratio 1 can mean either a compressed layer
        # or a pure sliding-window layer; the model tells them apart and passes
        # has_compressed explicitly (defaults to the V4 rule for back-compat).
        has_compressed = kwargs.pop("has_compressed", None)
        self.has_compressed = (compress_ratio > 1) if has_compressed is None else bool(has_compressed)
        # V4.1: official Attention has NO per-head RMSNorm on q after wq_b
        # (official-ref model.py forward); the V4 baseline applies one
        # (q_norm_without_weight / apply_dsa_q_rms). Models pass False to skip
        # it; default True keeps the V4 baseline path untouched.
        self.apply_q_head_rms = bool(kwargs.pop("apply_q_head_rms", True))
        self.kv_sharing_target_layer_name = kwargs.get("kv_sharing_target_layer_name", None)             if "kv_sharing_target_layer_name" in kwargs else None
        # Model-level buffer with the latest indexer topk indices. V4.1 layers
        # that neither own an indexer nor compress their own KV read it instead
        # of recomputing (official SharedAttentionRuntime.topk_idxs).
        self.topk_indices_buffer = kwargs.pop("topk_indices_buffer", None)
        # V4.1: latest candidate-block mask, published by the candidate source
        # layer (official SharedAttentionRuntime.candidates). Mirrored on the
        # model-level shared index runtime so non-owns_k index sources
        # (layers 24/28/32/36) can read what layer 20 published.
        self.candidates_buffer = None
        # R31 tear guard: host copy of the published pool (device tears
        # cannot reach host memory); readers upload fresh from this.
        self.candidates_host = None
        # V4.1 model-level hand-off object: owns_k layers publish an
        # index-K provider (their reference store) plus the candidates mask.
        self.shared_index_runtime = kwargs.pop("shared_index_runtime", None)
        self.softmax_scale = self.head_dim**-0.5
        self.support_fp8_attention = get_current_hardware_profile().supports(HardwareCapability.FP8_ATTENTION)

        # V4.1 official attention fp8 round-trip quant (patch 80): official
        # Attention.forward fake-quants the window KV right after RoPE
        # (act_quant(kv, 32, ue8m0, e4m3, inplace=True)) and the o activation
        # right after the inverse RoPE, per-32 along each o-group's hidden
        # dim. The engine previously kept both in bf16 end to end, which is a
        # documented part of the per-layer attention residual (L0 attnout
        # rel 1.69 / cos 0.952). Env-gated for A/B: master
        # V41_OFFICIAL_ATTN_QUANT, per-site V41_WINKV_FP8_RT / V41_OPROJ_FP8_RT
        # (each defaults to the master switch).
        _v41_master_rt = os.environ.get("V41_OFFICIAL_ATTN_QUANT", "0") == "1"
        self._v41_winkv_fp8_rt = os.environ.get("V41_WINKV_FP8_RT", "1" if _v41_master_rt else "0") == "1"
        self._v41_oproj_fp8_rt = os.environ.get("V41_OPROJ_FP8_RT", "1" if _v41_master_rt else "0") == "1"

        # MLA Args
        self.wq_a = kwargs["wq_a"]
        self.wq_b = kwargs["wq_b"]
        self.wkv = kwargs["wkv"]
        self.q_norm = kwargs["q_norm"]
        self.q_norm_without_weight = kwargs["q_norm_without_weight"]
        self.kv_norm = kwargs["kv_norm"]

        # CV wrapper: split wq_a/wkv/wq_b into quantize(Vector) + matmul(Cube)
        self.cv_wq_a = CVLinearWrapper(self.wq_a)
        self.cv_wkv = CVLinearWrapper(self.wkv)
        self.cv_wq_b = CVLinearWrapper(self.wq_b)

        self.indexer = kwargs.get("indexer")
        self.compressor = kwargs.get("compressor")
        self.swa_cache_layer = kwargs.get("swa_cache_layer")
        assert self.swa_cache_layer is not None

        self.wo_a = kwargs["wo_a"]
        self.wo_b = kwargs["wo_b"]

        self.eps = kwargs["eps"]

        self.attn_sink = kwargs["attn_sink"]

        ascend_config = get_ascend_config()
        self.multistream_dsv4_dsa_overlap = ascend_config.multistream_dsv4_dsa_overlap
        if self.multistream_dsv4_dsa_overlap and is_a5_bf16_kv_enabled(self.vllm_config):
            self.multistream_dsv4_dsa_overlap = False

    def _shared_source_meta_key(self, attn_layer_name: str) -> str | None:
        """Metadata key of the kv-source layer owning this shared layer's cache."""
        import re
        cfg = self.vllm_config.model_config.hf_config
        srcs = sorted(getattr(cfg, "kv_source_layer_ids", None) or [])
        if not srcs:
            return None
        m = re.search(r"layers\.(\d+)\.", attn_layer_name)
        if not m:
            return None
        idx = int(m.group(1))
        src = max((s for s in srcs if s <= idx), default=None)
        if src is None or src == idx:
            return None
        return re.sub(r"layers\.\d+\.", f"layers.{src}.", attn_layer_name, count=1)

    def _get_layer_metadata(
        self,
        attn_layer_name: str,
        attn_metadata: DSAMetadataDict,
    ) -> AscendDSALayerMetadata:
        assert self.swa_cache_layer is not None
        swa_metadata = attn_metadata[self.swa_cache_layer.prefix]
        attention_metadata = None
        compressor_metadata = None
        indexer_metadata = None

        if self.has_compressed:
            meta_key = attn_layer_name
            if meta_key not in attn_metadata:
                # Shared-compressed layer: its own name never got a spec
                # entry; read the nearest kv-source layer at-or-below it.
                meta_key = self._shared_source_meta_key(attn_layer_name) or attn_layer_name
            attention_metadata = attn_metadata[meta_key]
            state_metadata = None
            if self.compressor is not None and self.compressor.state_cache is not None:
                state_metadata = attn_metadata[self.compressor.state_cache.prefix]
            # V4.1 shared-compressed layers pass compressor=None; their
            # compressor_metadata is never consumed (they never pool KV).
            compressor_metadata = AscendCompressorMetadata(
                cache=attention_metadata,
                state=state_metadata,
            )
            if self.indexer is not None and getattr(self.indexer, "owns_k", True):
                assert self.indexer.k_cache is not None
                indexer_state_metadata = None
                indexer_compressor = getattr(self.indexer, "compressor", None)
                if indexer_compressor is not None and indexer_compressor.state_cache is not None:
                    indexer_state_metadata = attn_metadata[indexer_compressor.state_cache.prefix]
                indexer_cache_metadata = attn_metadata[self.indexer.k_cache.prefix]
                indexer_metadata = AscendIndexerMetadata(
                    compressor=AscendCompressorMetadata(
                        cache=indexer_cache_metadata,
                        state=indexer_state_metadata,
                    ),
                )

        return AscendDSALayerMetadata(
            attention=attention_metadata,
            swa=swa_metadata,
            compressor=compressor_metadata,
            indexer=indexer_metadata,
        )

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config=None,
        speculative_config=None,
        draft_attn_metadatas=None,
    ):
        # dsa does not need to update graph params
        pass

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # Attention impls are not walked by vllm's process_weights_after_loading
        # dispatcher (only LinearMethodBase subclasses are). OTP buffers are
        # allocated lazily on the first _forward_o_proj call, which always runs
        # before ACL graph capture (profiling run triggers it).
        pass

    def _forward_o_proj(self, o_proj_input: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        num_tokens = o_proj_input.shape[0]
        group_hidden_dim = o_proj_input.shape[1] * o_proj_input.shape[2] // self.n_local_groups
        o_proj_input = o_proj_input.view(num_tokens, self.n_local_groups, group_hidden_dim)
        if self._v41_oproj_fp8_rt:
            # official act_quant(o, 32, ue8m0, e4m3, inplace=True) before the
            # wo_a einsum: snap the o activation (post inverse-RoPE) to the
            # official fp8 grid per-32 along the group hidden dim, ahead of
            # the engine's dynamic-MX quantized o-proj below.
            from vllm_ascend.models.deepseek_v41.indexer import fp8_fake_quant_inplace_ue8m0

            fp8_fake_quant_inplace_ue8m0(o_proj_input)
        # A5 (Ascend950) uses an FP8-quantized o_proj path (dynamic MX quant
        # + quantized batch matmul). Preserve it as-is: it predates and is
        # orthogonal to the OTP / olora_tp paths below, so it must win first.
        use_a5_quant_o_proj = self.support_fp8_attention and _has_weight_scale(self.wo_a)
        if use_a5_quant_o_proj:
            o = o_proj_input
            o, swiglu_out_scale = torch_npu.npu_dynamic_mx_quant(o, dst_type=torch.float8_e4m3fn)
            o = torch_npu.npu_transpose_quant_batchmatmul(
                o,
                self.wo_a.weight,
                dtype=torch.bfloat16,
                bias=None,
                group_sizes=(0, 0, 32),
                x1_scale=swiglu_out_scale.view(torch.float8_e8m0fnu),
                x2_scale=self.wo_a.weight_scale.view(torch.float8_e8m0fnu),
                perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
            )
            o = o.reshape(num_tokens, -1)
            output[...] = self.wo_b(o)
        elif oproj_tp_enable():
            oproj_group = get_otp_group()
            oproj_tp_size = oproj_group.world_size
            if self.n_local_groups % oproj_tp_size != 0:
                raise ValueError(
                    "n_local_groups must be divisible by "
                    f"oproj_tensor_parallel_size, got {self.n_local_groups} "
                    f"and {oproj_tp_size}."
                )

            groups_per_rank = self.n_local_groups // oproj_tp_size

            o_proj_input = o_proj_input.view(num_tokens, oproj_tp_size, groups_per_rank, group_hidden_dim)
            # Pad to a static exchange size so the all_to_all / reduce_scatter
            # shapes are identical across all ACL graph buckets — variable
            # shapes desync the HCCL communicator during graph replay.
            # potential_max_tokens is computed once in the model runner __init__,
            # so reading it here is a cheap global lookup.
            exchange_num_tokens = get_potential_max_tokens()
            if exchange_num_tokens < num_tokens:
                raise ValueError(
                    "oproj static exchange capacity must cover local tokens, "
                    f"got {exchange_num_tokens} and {num_tokens}."
                )
            # Lazily allocate static send/recv buffers on first call. The
            # profiling run hits this path before ACL graph capture, so the
            # buffers exist and keep a stable device address across all later
            # capture/replay cycles (graph replay requires the same address
            # that was recorded at capture; new_zeros per call would desync
            # the HCCL operator).
            if not hasattr(self, "_oproj_send_buf"):
                buf_shape = (oproj_tp_size, exchange_num_tokens, groups_per_rank, group_hidden_dim)
                self._oproj_send_buf = torch.zeros(buf_shape, dtype=o_proj_input.dtype, device=o_proj_input.device)
                self._oproj_recv_buf = torch.empty_like(self._oproj_send_buf)
            send = self._oproj_send_buf
            recv = self._oproj_recv_buf
            # In-place fill into the address-stable buffer: zero the padding
            # tail, then copy the real tokens.
            send.zero_()
            send[:, :num_tokens].copy_(o_proj_input.transpose(1, 0))
            dist.all_to_all_single(recv.view(-1), send.view(-1), group=oproj_group.device_group)
            o_proj_input = recv.view(oproj_tp_size * exchange_num_tokens, groups_per_rank, group_hidden_dim)
            o_proj_input = torch_npu.npu_transpose_batchmatmul(
                o_proj_input,
                self.wo_a.weight,
                bias=None,
                scale=None,
                perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
                batch_split_factor=1,
            )
            o_proj_input = o_proj_input.reshape(oproj_tp_size * exchange_num_tokens, -1)
            o_proj_output = self.wo_b(o_proj_input)
            # reduce_scatter via a raw dist collective into an address-stable
            # static buffer. oproj_group.reduce_scatter is a list-based wrapper
            # that allocates per call, which desyncs the HCCL operator recorded
            # at capture during ACL graph replay — the same reason all_to_all
            # and the embedding TP path use raw dist + static buffers.
            if not hasattr(self, "_oproj_rs_out_buf"):
                self._oproj_rs_out_buf = torch.empty(
                    (exchange_num_tokens, o_proj_output.shape[-1]),
                    dtype=o_proj_output.dtype,
                    device=o_proj_output.device,
                )
            dist.reduce_scatter_tensor(self._oproj_rs_out_buf, o_proj_output, group=oproj_group.device_group)
            output[...] = self._oproj_rs_out_buf[:num_tokens]
        elif olora_tp_enable():
            o_proj_input = self.wo_a(o_proj_input)
            output[...] = self.wo_b(o_proj_input)
        else:
            # A5 BF16 wo_a is reshaped to [groups, hidden, rank] at load time,
            # matching the A3 layout expected by npu_transpose_batchmatmul.
            o_proj_input = torch_npu.npu_transpose_batchmatmul(
                o_proj_input,
                self.wo_a.weight,
                bias=None,
                scale=None,
                perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
                batch_split_factor=1,
            )
            o_proj_input = o_proj_input.reshape(num_tokens, -1)
            output[...] = self.wo_b(o_proj_input)
        return output

    def _prepare_caches_before_attention(
        self,
        layer_name: str,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: DSAMetadataDict,
    ) -> bool:
        """Prepare cache updates and report whether local writes can be skipped."""
        return False

    def _get_o_proj_input_shape(
        self,
        attn_metadata: DSAMetadataDict | None,
    ) -> tuple[int, int, int]:
        return (
            _EXTRA_CTX.num_tokens,
            self.n_local_heads,
            self.head_dim,
        )

    def forward(
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: tuple[torch.Tensor, ...] | None,
        attn_metadata: DSAMetadataDict,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        output_padded = output
        o_proj_input_shape = self._get_o_proj_input_shape(attn_metadata)
        if attn_metadata is None:
            # Profiling run: run o_proj on zero input so HCCL collectives are
            # captured by the ACL graph.  Non-OTP just zeros the output.
            if oproj_tp_enable():
                o_proj_input = hidden_states.new_zeros(o_proj_input_shape)
                self._forward_o_proj(o_proj_input, output)
            else:
                output.fill_(0)
            return output
        layer_metadata = self._get_layer_metadata(layer_name, attn_metadata)
        common_attn_metadata = layer_metadata.attention
        if common_attn_metadata is None:
            common_attn_metadata = layer_metadata.swa
        actual_tokens = common_attn_metadata.num_actual_tokens

        o_proj_input = hidden_states.new_zeros(o_proj_input_shape)
        assert kv_cache is not None, "kv_cache tensor tuple must be provided."
        wait_for_kv_layer_from_connector(layer_name)
        cache_is_prepared = self._prepare_caches_before_attention(
            layer_name,
            hidden_states,
            kv_cache,
            attn_metadata,
        )
        if actual_tokens == 0:
            output.zero_()
            notify_kv_cache_written(layer_name)
            maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))
            return output

        req_metadata = _require_req_metadata(common_attn_metadata)
        o_proj_input[:actual_tokens] = self._forward_attention(
            layer_name,
            hidden_states[:actual_tokens],
            kv_cache,
            layer_metadata,
            cache_is_prepared,
        )
        cos = req_metadata.cos[layer_name]
        sin = req_metadata.sin[layer_name]

        torch.ops._C_ascend.inplace_partial_rotary_mul(
            o_proj_input[:actual_tokens].unsqueeze(1),
            cos[:actual_tokens],
            -sin[:actual_tokens],
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        # o
        # R56: one-shot dump of L0's attention result BEFORE o_proj (after
        # inverse rope) — bisects window-attention vs o_proj for the eager
        # prefill L0 divergence.
        try:
            from vllm_ascend.models.deepseek_v41.model import _V41_DUMP as _D
            if (_D.get("active") and not _D.get("attn0pre_done")
                    and layer_name.endswith(".layers.0.self_attn.attn")):
                _D["attn0pre_done"] = True
                import os as _os

                import torch as _t
                _t.save(
                    {"tensor": o_proj_input[:actual_tokens].detach().cpu(),
                     "shape": tuple(o_proj_input.shape),
                     "dtype": str(o_proj_input.dtype)},
                    _os.path.join(
                        _D["dir"],
                        f"L0_pid{_os.getpid()}_r"
                        f"{_os.environ.get('VLLM_DP_RANK', '0')}"
                        f"_attn_pre_oproj.pt"))
        except Exception:  # noqa: BLE001 - probe must never kill serving
            pass
        self._forward_o_proj(o_proj_input, output)
        maybe_save_kv_layer_to_connector(layer_name, list(kv_cache))

        _v41_prefill_dump = (
            __import__("os").environ.get("V41_DUMP_PREFILL", "0") == "1"
        )
        if (actual_tokens <= 4 or _v41_prefill_dump) and _v41_dbg_enabled("dump"):
            try:
                _total = _v41_layer_total(req_metadata)
                if _v41_dbg_total_ok(_total) and _v41_dbg_layer_ok(layer_name):
                    _v41_dbg_save(
                        {
                            "attnin": hidden_states[:actual_tokens].detach().cpu(),
                            "attnout": output[:actual_tokens].detach().cpu(),
                            "seq_lens": req_metadata.seq_lens[: req_metadata.num_actual_reqs or req_metadata.seq_lens.shape[0]].detach().cpu().tolist(),
                        },
                        _v41_dbg_path("V41_DUMP_DECODE", layer_name, _total, "lyr"),
                    )
            except Exception as _e:  # noqa: BLE001 - debug hook must never kill serving
                import traceback as _tb
                print(f"[V41_DBG_SAVE_ERR] {type(_e).__name__}: {_e}\n"
                      + "\n".join(_tb.format_exc().splitlines()[-4:]),
                      flush=True)

        return output_padded

    def _mla_prolog_single_stream(
        self,
        hidden_states,
        cos,
        sin,
        swa_kv_cache,
        slot_mapping,
        write_swa_cache=True,
    ):
        """Run the MLA prolog on the current stream."""
        share_hs_quant = _is_w8a8_dynamic(self.wq_a) and _is_w8a8_dynamic(self.wkv)
        if share_hs_quant:
            hs_int8, hs_pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
            q_a = torch_npu.npu_quant_matmul(
                hs_int8,
                self.wq_a.weight,
                self.wq_a.weight_scale,
                pertoken_scale=hs_pertoken_scale,
                bias=self.wq_a.bias,
                output_dtype=hidden_states.dtype,
            )
        else:
            q_a = self.wq_a(hidden_states)
        if hidden_states.shape[0] <= 64:
            _v41_r57_dump("q_a", q_a)

        # q
        if _is_w8a8_dynamic(self.wq_b):
            qr, qr_pertoken_scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(
                q_a, self.q_norm.weight, epsilon=self.eps
            )
            if hidden_states.shape[0] <= 64:
                _v41_r57_dump("qr", qr)
            q = torch_npu.npu_quant_matmul(
                qr,
                self.wq_b.weight,
                self.wq_b.weight_scale,
                pertoken_scale=qr_pertoken_scale,
                bias=self.wq_b.bias,
                output_dtype=hidden_states.dtype,
            ).unflatten(-1, (self.n_local_heads, self.head_dim))
        else:
            qr = self.q_norm(q_a)
            if hidden_states.shape[0] <= 64:
                _v41_r57_dump("qr", qr)
            q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
            if hidden_states.shape[0] <= 64:
                _v41_r57_dump("q_preb", q)
            qr_pertoken_scale = None
        if self.apply_q_head_rms:
            q = DeviceOperator.apply_dsa_q_rms(q, self.eps, self.q_norm_without_weight)

        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        if hidden_states.shape[0] <= 64:
            _v41_r57_dump("q_rope", q)
        # win kv & tok_dis
        if write_swa_cache:
            _kp = self._v41_kvprobe_get(hidden_states) \
                if _v41_layer_io_dir() is not None else None
            if share_hs_quant:
                kv = torch_npu.npu_quant_matmul(
                    hs_int8,
                    self.wkv.weight,
                    self.wkv.weight_scale,
                    pertoken_scale=hs_pertoken_scale,
                    bias=self.wkv.bias,
                    output_dtype=hidden_states.dtype,
                )
            else:
                kv = self.wkv(hidden_states)
            kv = self.kv_norm(kv)
            assert self.rope_head_dim is not None
            kv = kv.view(-1, 1, self.nope_head_dim + self.rope_head_dim)
            if _kp is not None:
                _t2 = min(kv.shape[0], 2)
                _kp["pre"][:_t2] = kv[:_t2].reshape(_t2, -1)

            torch.ops._C_ascend.inplace_partial_rotary_mul(
                kv.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[self.nope_head_dim, self.head_dim],
            )
            if _kp is not None:
                _kp["rot"][:_t2] = kv[:_t2].reshape(_t2, -1)
            if self._v41_winkv_fp8_rt:
                # official act_quant(kv, 32, ue8m0, e4m3, inplace=True): the
                # window KV goes through the fp8 round trip after RoPE and
                # before the ring-cache write, so the cache (and this step's
                # attention) sees exactly the values official feeds
                # sparse_attn.
                from vllm_ascend.models.deepseek_v41.indexer import fp8_fake_quant_inplace_ue8m0

                kv = fp8_fake_quant_inplace_ue8m0(
                    kv, probe=_kp if _kp is not None else None)
            if _kp is not None:
                _kp["post"][:_t2] = kv[:_t2].reshape(_t2, -1)

            if hidden_states.shape[0] <= 64:
                _v41_r57_dump("kv_rt", kv)
            get_dsa_attn_kv_plan(self.vllm_config).dsa_kv_compress_scatter(
                swa_kv_cache,
                kv,
                slot_mapping,
            )

        return q, qr, qr_pertoken_scale

    def _v41_kvprobe_get(self, sample: torch.Tensor):
        """Persistent per-layer kv probes (first-divergence experiment).

        Three snapshots of the decode kv on its way to the SWA cache:
        pre = post matmul+norm (pre-RoPE), rot = post-RoPE (pre-quant),
        post = post fp8 fake-quant (the tensor handed to the scatter).
        Plus the quant-internal stages of token 0 (review-9 bisect):
        qamax/qscale/qq/qout record fp8_fake_quant_inplace_ue8m0's own
        intermediates so each stage can be checked against an independent
        reference computed from the captured `rot` row.
        Plain device copies into persistent buffers, created on the first
        (eager warmup) call so later replays only re-record the writes.
        """
        kp = getattr(self, "_v41_kvprobe", None)
        if kp is None:
            w = self.nope_head_dim + self.rope_head_dim
            ngrp = w // 32  # fp8 fake-quant block count of token 0
            kp = {n: torch.zeros(2, w, dtype=torch.bfloat16,
                                 device=sample.device)
                  for n in ("pre", "rot", "post")}
            for _qn in ("qamax", "qdiv", "qexp", "qscale"):
                kp[_qn] = torch.zeros(ngrp, 1, dtype=torch.float32,
                                      device=sample.device)
            kp["qq"] = torch.zeros(ngrp, 32, dtype=torch.float32,
                                   device=sample.device)
            kp["qout"] = torch.zeros(ngrp, 32, dtype=torch.float32,
                                     device=sample.device)
            self._v41_kvprobe = kp
        return kp

    def _mla_prolog_multistream(self, hidden_states, cos, sin, swa_kv_cache, slot_mapping, is_prefill=False):
        """3-block multi-stream: 3-stage CV parallel + serial tail

        Block partition (V: Vector, C: Cube, AIV: AI Vector):
          Part1: q_quant[V] -> q_a_down[C]  ||  kv_quant[V]
          Part2: q_norm[V] + q_b_quant[V]  ||  kv_matmul[C]
          Part3: q_b_matmul[C]             ||  kv_norm[V] + rope[V] + scatter[AIV]
          Tail:  q_rms[V] + rope[V] (wait for auxiliary stream to complete)

        Each stream's data is self-contained; no cross-stream sync is needed between blocks.
        Only the tail wait_stream ensures scatter is complete.
        """
        main_stream = torch.npu.current_stream()
        aux_stream = dsv4_dsa_overlap_stream()

        is_w8a8 = _is_w8a8_dynamic(self.wq_b)

        # Part1: q_quant[V] -> q_a_down[C]  ||  kv_quant[V]
        # When wq_a and wkv have the same quant_method and the same
        # communication status, their quantize() outputs are equivalent.
        # Share the result instead of calling quantize() twice on the same input.
        # - W8A8 no-comm: saves one npu_dynamic_quant (full-tensor read + absmax).
        # - W4A8 no-comm: saves one no-op pass-through (kernel launch + ref).
        # - TP comm: both return (hidden_states, None); shareable when custom_op
        #   types match (same communication path).
        share_quant = (
            type(self.cv_wq_a._quant_method) is type(self.cv_wkv._quant_method)
            and self.cv_wq_a._has_communication == self.cv_wkv._has_communication
        )
        e_kv_quant_done = None
        if share_quant:
            q_quant, q_pertoken_scale = self.cv_wq_a.quantize(hidden_states)
            kv_quant, kv_pertoken_scale = q_quant, q_pertoken_scale
        else:
            q_quant, q_pertoken_scale = self.cv_wq_a.quantize(hidden_states)
            e_q_quant_done = main_stream.record_event()
            with npu_stream_switch(aux_stream, enabled=True):
                torch.npu.current_stream().wait_event(e_q_quant_done)
                kv_quant, kv_pertoken_scale = self.cv_wkv.quantize(hidden_states)
                e_kv_quant_done = torch.npu.current_stream().record_event()

        wq_a_result = self.cv_wq_a.matmul(q_quant, q_pertoken_scale)

        # Part2: q_norm[V] + q_b_quant[V]  ||  kv_matmul[C]
        e_part2_start = main_stream.record_event()
        if e_kv_quant_done is not None:
            main_stream.wait_event(e_kv_quant_done)

        with npu_stream_switch(aux_stream, enabled=True):
            torch.npu.current_stream().wait_event(e_part2_start)
            kv = self.cv_wkv.matmul(kv_quant, kv_pertoken_scale)
            e_kv_matmul_done = torch.npu.current_stream().record_event()

        if is_prefill:
            qr = self.q_norm(wq_a_result)
            q_b_quant, q_b_scale = self.cv_wq_b.quantize(qr)
            qr_pertoken_scale = None
        elif is_w8a8:
            qr, qr_pertoken_scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(
                wq_a_result, self.q_norm.weight, epsilon=self.eps
            )
            q_b_quant, q_b_scale = qr, qr_pertoken_scale
        else:
            qr = self.q_norm(wq_a_result)
            q_b_quant, q_b_scale = qr, None
            qr_pertoken_scale = None

        # Part3: q_b_matmul[C]  ||  kv_norm[V] + rope[V] + scatter[AIV]
        e_part3_start = main_stream.record_event()
        # kv_matmul and q_b_matmul are both Cube ops. Ensure kv_matmul (launched on
        # aux_stream) completes before q_b_matmul starts so they do not contend for
        # the Cube units. kv_norm (Vector) follows kv_matmul on aux_stream and is
        # unaffected as it overlaps with q_b_matmul.
        main_stream.wait_event(e_kv_matmul_done)

        with npu_stream_switch(aux_stream, enabled=True):
            torch.npu.current_stream().wait_event(e_part3_start)
            kv = self.kv_norm(kv)
            assert self.rope_head_dim is not None
            kv = kv.view(-1, 1, self.nope_head_dim + self.rope_head_dim)
            torch.ops._C_ascend.inplace_partial_rotary_mul(
                kv.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[self.nope_head_dim, self.head_dim],
            )
            if self._v41_winkv_fp8_rt:
                # official window-KV fp8 round trip (see _mla_prolog); runs on
                # the aux stream between RoPE and the scatter.
                from vllm_ascend.models.deepseek_v41.indexer import fp8_fake_quant_inplace_ue8m0

                # NOTE: the inplace round trip's mutation is DROPPED under
                # npugraph_ex FX (graph rows get UNQUANTIZED values); the
                # post-replay drain re-quantizes the exported row eagerly
                # before writing the pool (see v41_io_drain_all), so replay
                # #2 reads fp8-rounded rows exactly like eager. Out-of-place
                # variants of the quant broke the whole region's lowering
                # (value collapsed) — keep this exact call form.
                kv = fp8_fake_quant_inplace_ue8m0(kv)
            get_dsa_attn_kv_plan(self.vllm_config).dsa_kv_compress_scatter(swa_kv_cache, kv, slot_mapping)

        if is_prefill:
            q = self.cv_wq_b.matmul(q_b_quant, q_b_scale).unflatten(-1, (self.n_local_heads, self.head_dim))
        elif is_w8a8:
            q = torch_npu.npu_quant_matmul(
                q_b_quant,
                self.wq_b.weight,
                self.wq_b.weight_scale,
                pertoken_scale=q_b_scale,
                bias=self.wq_b.bias,
                output_dtype=hidden_states.dtype,
            ).unflatten(-1, (self.n_local_heads, self.head_dim))
        else:
            q = self.cv_wq_b.matmul(q_b_quant, q_b_scale).unflatten(-1, (self.n_local_heads, self.head_dim))

        # Serial tail: wait for auxiliary stream then execute q_rms[V] + rope[V]
        main_stream.wait_stream(aux_stream)

        if self.apply_q_head_rms:
            q = DeviceOperator.apply_dsa_q_rms(q, self.eps, self.q_norm_without_weight)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        return q, qr, qr_pertoken_scale

    def _maybe_update_compressed_caches_and_select_topk(
        self,
        layer_name: str,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        layer_metadata: AscendDSALayerMetadata,
        qr_pertoken_scale: torch.Tensor | None,
        compress_kv_cache: torch.Tensor,
        state_cache: torch.Tensor,
        write_cache: bool = True,
    ) -> torch.Tensor | None:
        """Update compressed caches and return Indexer top-k indices."""
        compressor = self.compressor
        assert layer_metadata.compressor is not None

        # V4.1 indexers carry is_v41_indexer=True (deepseek_v41/indexer.py);
        # the V4 baseline indexer sets it to False. Unknown indexers default
        # onto the V4.1 reference bridge per the bridge contract.
        is_v41_indexer = self.indexer is not None and getattr(self.indexer, "is_v41_indexer", True)

        if (
            self.indexer is not None
            and layer_metadata.indexer is not None
            and not is_v41_indexer
        ):
            # V4 c4 layers own an indexer.
            assert compressor is not None

            def compute_attention_compressed_kv() -> tuple[torch.Tensor, torch.Tensor]:
                return compressor(
                    hidden_states=hidden_states,
                    state_cache=state_cache,
                    metadata=layer_metadata.compressor,
                )

            def scatter_attention_compressed_kv(
                compressed_kv: torch.Tensor,
                compress_slot_mapping: torch.Tensor,
            ) -> None:
                if compressed_kv.shape[0] > 0:
                    get_dsa_attn_kv_plan(self.vllm_config).dsa_kv_compress_scatter(
                        compress_kv_cache,
                        compressed_kv,
                        compress_slot_mapping,
                    )

            overlap_plan = IndexerOverlapPlan(
                compute_attention_compressed_kv=compute_attention_compressed_kv,
                scatter_attention_compressed_kv=scatter_attention_compressed_kv,
                aux_stream=dsv4_dsa_overlap_stream() if self.multistream_dsv4_dsa_overlap else None,
            )
            return self.indexer(
                hidden_states=hidden_states,
                qr=qr,
                kv_cache=kv_cache,
                metadata=layer_metadata.indexer,
                overlap_plan=overlap_plan,
                layer_name=layer_name,
                qr_pertoken_scale=qr_pertoken_scale,
                write_cache=write_cache,
            )

        if self.indexer is not None:
            if is_v41_indexer:
                # V4.1 index-source layers: owns_k (kv_source 2/8/14/20) pool
                # their own latents; layers 24/28/32/36 score the shared
                # index K cache the most recent owner published.
                return self._v41_reference_indexer_topk(
                    layer_name=layer_name,
                    hidden_states=hidden_states,
                    qr=qr,
                    layer_metadata=layer_metadata,
                    compress_kv_cache=compress_kv_cache,
                    state_cache=state_cache,
                    write_cache=write_cache,
                    kv_cache=kv_cache,
                )
            # Unreachable for the V4 baseline (its indexers always carry
            # k-cache metadata); kept as a loud guard.
            raise NotImplementedError(
                "indexer without k-cache metadata has no top-k path; "
                "see deepseek_v41/indexer.py"
            )

        if compressor is None:
            # V4.1 shared compressed layer: neither compresses its own KV nor
            # selects top-k; reuse what the owning index source published.
            return self._get_shared_topk_indices(hidden_states.shape[0])

        if not write_cache:
            return None
        compressed_kv, compress_slot_mapping = compressor(
            hidden_states=hidden_states,
            state_cache=state_cache,
            metadata=layer_metadata.compressor,
        )
        if compressed_kv.shape[0] > 0:
            get_dsa_attn_kv_plan(self.vllm_config).dsa_kv_compress_scatter(
                compress_kv_cache,
                compressed_kv,
                compress_slot_mapping,
            )
        return None

    def _v41_indexer_bt(self, indexer_md, req_md, B, dev):
        """Indexer-cache block table; falls back to the request table when
        the dummy capture metadata carries none."""
        bt = None
        src = "none"
        try:
            _rm = indexer_md.compressor.cache.req_metadata
            bt = getattr(_rm, "block_table", None) if _rm is not None else None
            if bt is not None:
                src = "indexer"
        except AttributeError:
            bt = None
        if bt is None:
            bt = getattr(req_md, "block_table", None)
            if bt is not None:
                src = "compressor"
        if bt is None:
            raise RuntimeError(
                "V4.1 device path: no block table available (indexer and "
                "compressor metadata both carry none) — capture metadata "
                "preparation for the indexer cache group must be checked "
                "(aclgraph for_capture path)")
        return bt[:B].to(dev).long()

    def _v41_pool_identity(self, req_md, B, dev):
        """Device-pool state row ids for this step's B requests: the engine
        request slot (persist channel, refreshed by the eager prepare hook
        EVERY step incl. prefill) when available, else the first block id.
        One resolver for both the decode device path and the eager seeding
        so the two can never key rows differently."""
        _p = getattr(req_md, "v41_persist", None)
        if _p is not None:
            return _p["req_slot"][:B].long()
        return req_md.block_table[:B, 0].long()

    def _v41_device_pool_state(self, layer_name, ratio, dev):
        """Per-layer DevicePoolState keyed by ENGINE REQUEST SLOT ids."""
        st = getattr(self, "_v41_dev_state", None)
        if st is None:
            self._v41_dev_state = st = {}
        layer_state = st.get(layer_name)
        if layer_state is None:
            from vllm_ascend.models.deepseek_v41.indexer import DevicePoolState

            layer_state = {"pool": DevicePoolState(
                int(self.vllm_config.scheduler_config.max_num_seqs), ratio,
                int(self.compressor.wkv.weight.shape[0]), dev)}
            st[layer_name] = layer_state
        return layer_state["pool"]

    def _v41_seed_device_pool_carry(self, layer_name, indexer, req_md, B,
                                    first_blocks, seq_lens_cpu):
        """Prefill -> graph-decode hand-off: copy the reference pool's
        trailing PARTIAL-group carry (official kv_state/score_state) into
        this layer's device pool rows, so the first decode steps complete
        the group over exactly the tokens eager pooling saw. Without this,
        a request entering decode mid-group pools stale state slots."""
        if __import__("os").environ.get("V41_DEVICE_INDEXER", "0") != "1":
            return
        if indexer.ref_pool is None or indexer.ref_pool.kv_state is None:
            return
        ratio = indexer.compress_ratio
        dev = req_md.block_table.device
        pool = self._v41_device_pool_state(layer_name, ratio, dev)
        if pool.kv is None:
            return
        slots = self._v41_pool_identity(req_md, B, dev).detach().cpu().tolist()
        for i in range(B):
            rem = int(seq_lens_cpu[i]) % ratio
            if rem == 0:
                continue
            req_key = int(first_blocks[i])
            row = indexer.ref_pool.row_of_req.get(req_key)
            if row is None:
                continue
            if indexer.ref_pool.row_len.get(req_key, -1) != int(seq_lens_cpu[i]):
                # the reference row belongs to a different schedule on this
                # block id (eviction/prefix mix): its carry is foreign
                continue
            pool.seed_carry(
                int(slots[i]),
                indexer.ref_pool.kv_state[row, :rem],
                indexer.ref_pool.score_state[row, :rem], rem)

    def _v41_device_indexer_topk(
        self, layer_name, hidden_states, qr, req_md, indexer_md, kv_cache,
        indexer, owns_k, compress_kv_cache=None,
    ) -> torch.Tensor:
        """Patch 85 decode indexer path — verified conventions (see
        work/tests/cache_roundtrip.py, ALL PASS with the REAL ops):
        page = 16384B key region + 512B scale region (ONE fp32/row);
        epilog takes FLAT int32 slots page*rows_pp+row. Official visibility:
        the group completing THIS step participates THIS step (akl =
        end_pos//ratio = seq_lens//ratio for decode); its row is scattered
        BEFORE selection. Pool state rows keyed by the ENGINE REQUEST SLOT
        (persist channel; first block id fallback — shared prefixes collide
        on first blocks between concurrent requests).
        """
        from vllm_ascend.device.device_op import DeviceOperator

        dev = hidden_states.device
        B = hidden_states.shape[0]
        ratio = indexer.compress_ratio
        _, ik_cache, isc_cache, ifc_cache = DeviceOperator.unpack_dsa_indexer_kv_cache(kv_cache)
        _idx_bt = None
        if __import__("os").environ.get("V41_GRAPH_PTRDBG", "0") == "1":
            import os as _os
            _os.makedirs("/ws/evidence/p85run1/ptrdbg", exist_ok=True)
            _f = f"/ws/evidence/p85run1/ptrdbg/{layer_name.split('.')[-1]}.txt"
            _cap = torch.npu.is_current_stream_capturing() if hasattr(torch.npu, "is_current_stream_capturing") else False
            with open(_f, "a") as _fh:
                _fh.write(f"cap={_cap} compiling={torch.compiler.is_compiling()} "
                          f"req_bt_ptr={req_md.block_table.data_ptr() if req_md.block_table is not None else None} "
                          f"req_sl_ptr={req_md.seq_lens.data_ptr()} "
                          f"hidden_rows={hidden_states.shape[0]}\n")
        _persist = getattr(req_md, "v41_persist", None)
        # The indexer cache's block table MUST come from the persist channel
        # under graph replay: req_metadata.block_table is a per-step tensor
        # (new storage each build) whose capture-time instance is frozen into
        # the graph — replayed gathers would read capture-dummy pages. The
        # persist set is refreshed by every eager build before each replay.
        _idx_bt = _persist["ori_bt"] if _persist is not None else None
        if owns_k:
            if _idx_bt is None:
                try:
                    _rm = indexer_md.compressor.cache.req_metadata
                    _idx_bt = getattr(_rm, "block_table", None) if _rm is not None else None
                except AttributeError:
                    _idx_bt = None
            if self.shared_index_runtime is not None and _idx_bt is not None:
                # publish caches AND the indexer block table (persist buffer
                # under graph, per-step tensor in eager); non-owns_k index
                # sources score the owner's cache and carry no indexer
                # metadata of their own
                self.shared_index_runtime.dev_handles = (
                    ik_cache, isc_cache, ifc_cache, _idx_bt)
        elif self.shared_index_runtime is not None and getattr(
            self.shared_index_runtime, "dev_handles", None
        ) is not None:
            # non-owns_k index sources: score the OWNER's published caches
            # and block table (their own kv_cache tuple belongs to a
            # non-indexer attention group and must not be unpacked here)
            ik_cache, isc_cache, ifc_cache, _pub_bt = (
                self.shared_index_runtime.dev_handles)
            if _idx_bt is None:
                _idx_bt = _pub_bt
        # pool-state row identity: engine request slot (persist channel) or
        # first block id — one resolver shared with the eager seeding
        pool_slots = self._v41_pool_identity(req_md, B, dev)

        if owns_k:
            pool = self._v41_device_pool_state(layer_name, ratio, dev)
            wgate = (self.compressor.wgate.weight
                     if self.compressor.wgate is not None else None)
            # pool_decode returns the RMSNorm-ed latent (official
            # Compressor.forward: pooling THEN self.norm) — the same stage
            # ReferenceCompressorPool.pool returns; index keys AND cmp KV
            # both consume it.
            latent, complete, group = pool.pool_decode(
                hidden_states, self.compressor.wkv.weight, wgate,
                self.compressor.norm.weight, indexer.eps,
                req_md.start_pos[:B].long(), pool_slots)
            # quant=None follows the indexer's fake_quant_index_kv (official
            # fp4 fake-quant on indexer queries and keys); the previous
            # quant=False silently dropped an official computation.
            keys = indexer.compute_index_key(latent, (group * ratio).to(dev))

            # CONSTANT-SHAPE scatter (graph-safe, no host read): every row
            # scatters. Rows WITH a completion scatter the fresh key to its
            # group slot. Rows WITHOUT one scatter their key to the row of
            # the group the CURRENT token sits in (start_pos//ratio) — that
            # group is not yet visible (akl = seq//ratio excludes it), so
            # overwriting it cannot affect this step's selection; it gets
            # rewritten by the real key when the group completes.
            rows_pp = int(ik_cache.shape[1])
            idx_bt = (_idx_bt if _idx_bt is not None
                      else self._v41_indexer_bt(indexer_md, req_md, B, dev))
            g_sel = torch.where(
                complete.to(dev), group.to(dev),
                (req_md.start_pos[:B].long().to(dev) // ratio))
            col = (g_sel // rows_pp).clamp(max=idx_bt.shape[1] - 1)
            page = idx_bt[:B].gather(1, col.unsqueeze(1)).squeeze(1)
            slots_f = (page * rows_pp + g_sel % rows_pp).to(torch.int32).contiguous()
            keys_f = keys.to(dev).contiguous()
            DeviceOperator.indexer_quant_scatter_part1(
                keys_f, ik_cache,
                ifc_cache if ifc_cache is not None else ik_cache, slots_f)


            # compressed-KV scatter: the SAME cmp_kv_from_latent helper the
            # eager reference write uses (normed latent -> rope -> bf16 ->
            # fp4-e4m3scale). Only the graph branch passes the cmp pool (the
            # eager dual-run's reference impl writes the same slots with the
            # same helper right after); under replay the reference never
            # runs, so without this the decode steps' new groups would leave
            # the cmp pool unwritten and the sparse attention reads
            # uninitialized rows.
            if compress_kv_cache is not None:
                from vllm_ascend.models.deepseek_v41.indexer import cmp_kv_from_latent

                cmp_kv = cmp_kv_from_latent(indexer, latent, (group * ratio).to(dev))
                # cmp pool geometry: one row per compressed position, slots from
                # the request block table (same page ids as the indexer pool)
                cmp_rpp = int(compress_kv_cache.shape[1])
                g_cmp = torch.where(
                    complete.to(dev), group.to(dev),
                    (req_md.start_pos[:B].long().to(dev) // ratio))
                ccol = (g_cmp // cmp_rpp).clamp(max=idx_bt.shape[1] - 1)
                cpage = idx_bt[:B].gather(1, ccol.unsqueeze(1)).squeeze(1)
                cslots = (cpage * cmp_rpp + g_cmp % cmp_rpp).to(torch.int32).contiguous()
                get_dsa_attn_kv_plan(self.vllm_config).dsa_kv_compress_scatter(
                    compress_kv_cache,
                    cmp_kv.contiguous().view(B, 1, cmp_kv.shape[-1]),
                    torch.stack([cpage.to(torch.int32),
                                 (g_cmp % cmp_rpp).to(torch.int32)], dim=-1).contiguous(),
                )

        _seq_src = _persist["seq"] if _persist is not None else req_md.seq_lens
        akl = (_seq_src[:B].long() // ratio).clamp(min=0).to(torch.int32)

        # NOTE: no host reads here — .item()/bool(tensor) inside NPU graph
        # capture raises "stream is captured". The select path below is
        # value-agnostic: akl==0 (capture dummy, seq_len==1) simply masks
        # every position and the validity gate emits all -1, the same
        # tensor the eager max_cmp==0 early exit produces.

        if __import__("os").environ.get("V41_GRAPH_STEPDBG", "0") == "1":
            dbg = getattr(self, "_v41_stepdbg", None)
            if dbg is None:
                dbg = {
                    "step": torch.zeros(1, dtype=torch.int64, device=dev),
                    "akl": torch.zeros(8, dtype=torch.int32, device=dev),
                    "seq": torch.zeros(8, dtype=torch.int64, device=dev),
                    "bt": torch.zeros(8, dtype=torch.int64, device=dev),
                    "topk": torch.full((8,), -2, dtype=torch.int32, device=dev),
                }
                self._v41_stepdbg = dbg
            if _persist is not None:
                # register into the builder-owned persist set so the eager
                # build() drain reads REAL graph-side tensors (the previous
                # drain read builder-local attrs the impl never wrote). No
                # polling thread: .item() from a thread during an active
                # capture breaks stream capture (507034).
                _persist["stepdbg"][layer_name] = dbg
            # captured-side updates (pure device ops, replay-safe)
            dbg["step"] += 1
            _b = min(B, 8)
            dbg["akl"][:_b] = akl[:_b]
            dbg["seq"][:_b] = _seq_src[:_b].long()

        _sel = __import__("os").environ.get("V41_DEV_SELECT", "torch")
        if _sel == "1" or _sel == "skip":
            ref_sel = self.topk_indices_buffer[:B]
            return ref_sel.unsqueeze(1).contiguous()
        if _sel in ("constm1", "constz", "constuniq"):
            # value-bisect ladder: const [0,1,2 cycled] was CLEAN (cycle40);
            # constm1 [0,1,2]+(-1)x509 TRAPPED (cycle51). constz pads with 0
            # instead of -1; constuniq uses 64 distinct values cycled.
            # Separates "-1 tail" from "pad value / distinctness" effects.
            _cm = getattr(self, "_v41_constm1_buf", None)
            if _cm is None or _cm.shape[-1] != indexer.index_topk or _cm.device != dev:
                if _sel == "constz":
                    _t = torch.zeros(1, indexer.index_topk, dtype=torch.int32, device=dev)
                    _t[0, :3] = torch.arange(3, device=dev, dtype=torch.int32)
                elif _sel == "constuniq":
                    _t = (torch.arange(indexer.index_topk, device=dev) % 64).to(
                        torch.int32).view(1, -1)
                else:
                    _t = torch.full((1, indexer.index_topk), -1,
                                    dtype=torch.int32, device=dev)
                    _t[0, :3] = torch.arange(3, device=dev, dtype=torch.int32)
                _cm = _t.contiguous()
                self._v41_constm1_buf = _cm
            out_cm = _cm.expand(B, 1, -1).contiguous()
            if self.topk_indices_buffer is not None:
                self.topk_indices_buffer[:B].copy_(out_cm.reshape(B, -1))
            return out_cm
        if _sel == "const":
            # graph-trap bisect: return a CONSTANT, always-valid index set
            # (positions 0..2 cycled) from a stable buffer instead of the
            # computed selection. If the replay trap disappears, the killer
            # is the in-graph selection path's replayed output.
            _ct = getattr(self, "_v41_const_topk_buf", None)
            if (
                _ct is None or _ct.shape[-1] != indexer.index_topk
                or _ct.device != dev
            ):
                _ct = (torch.arange(indexer.index_topk, device=dev) % 3).to(
                    torch.int32).view(1, 1, -1).contiguous()
                self._v41_const_topk_buf = _ct
            out_ct = _ct.expand(B, 1, -1).contiguous()
            if self.topk_indices_buffer is not None:
                self.topk_indices_buffer[:B].copy_(out_ct.reshape(B, -1))
            return out_ct

        rows_pp = int(ik_cache.shape[1])
        # graph-safe: fixed gather width from the table; extra rows are
        # masked by akl below (no host reads)
        bt_dev = (_idx_bt if _idx_bt is not None
                  else self._v41_indexer_bt(indexer_md, req_md, B, dev))
        bt_dev = bt_dev[:B].to(dev).long()
        npages = int(bt_dev.shape[1])
        if _sel.startswith("w") and _sel[1:].isdigit():
            # bisect: clamp the table read width (W = npages*rows_pp) to test
            # whether the in-graph topk breakage is size-dependent
            npages = min(npages, int(_sel[1:]))
            bt_dev = bt_dev[:, :npages]
        dbg = getattr(self, "_v41_stepdbg", None)
        if dbg is not None:
            _b = min(B, 8)
            dbg["bt"][:_b] = bt_dev[:_b, 0]
        k8 = ik_cache.view(torch.uint8) if ik_cache.dtype == torch.float8_e4m3fn else ik_cache
        if _sel == "kconst":
            # bisect: replace the dynamic cache read (index_select on the
            # fp8/uint8 view with persist-driven page ids) with a stable
            # constant K; keep q/score/topk/argsort downstream intact
            _kc = getattr(self, "_v41_kconst_buf", None)
            if _kc is None or _kc.shape[1] != npages * rows_pp:
                _kc = (torch.arange(npages * rows_pp, device=dev, dtype=torch.float32)
                       .remainder(128).unsqueeze(-1).repeat(1, ik_cache.shape[-1])
                       .unsqueeze(0).expand(B, -1, -1).contiguous() * 0.001)
                self._v41_kconst_buf = _kc
            k_rows = _kc
        else:
            k_pages = k8.index_select(0, bt_dev[:, :npages].reshape(-1)).reshape(
                B, npages * rows_pp, ik_cache.shape[-1])
            if k_pages.dtype == torch.uint8:
                k_pages = k_pages.view(torch.float8_e4m3fn)
            s_rows = isc_cache.index_select(0, bt_dev[:, :npages].reshape(-1)).reshape(
                B, npages * rows_pp)
            W = npages * rows_pp
            k_rows = k_pages.float() * s_rows.float().unsqueeze(-1)
        W = npages * rows_pp

        # quant default follows fake_quant_index_kv (official fp4 fake-quant
        # on the indexer query; dropping it changed selection semantics)
        q = indexer.compute_query(qr, req_md.start_pos[:B].long())
        # under torch.compile the linear may return a bare tensor instead
        # of the eager (out,) tuple — index[0] would silently take ROW 0
        _w_out = indexer.weights_proj(hidden_states)
        _wp = _w_out[0] if isinstance(_w_out, tuple) else _w_out
        w = _wp.reshape(B, -1) * (
            indexer.softmax_scale * indexer.n_heads ** -0.5)
        score = torch.einsum("bhd,bwd->bhw", q.float(), k_rows).relu_()
        score = (score * w.float().unsqueeze(-1)).sum(dim=1)
        pos = torch.arange(W, device=dev).view(1, -1)
        score = score.masked_fill(pos >= akl.view(-1, 1), -float("inf"))

        # Official hierarchical rule (review fix): the candidate source layer
        # (L20) first selects candidate_topk_blocks blocks by their best
        # score (select_candidate_blocks math); every later index source may
        # only pick positions inside those blocks. The eager reference
        # implements this (compute_candidates -> shared runtime -> non-owns_k
        # masked_fill) but the device path did NOT — a confirmed semantic gap.
        # Graph-safe formulation: per-block amax, pin the block holding the
        # query's newest visible position, graph_safe_topk over blocks,
        # expand the keep mask to positions via gather. The mask lives in a
        # PERSISTENT model-level buffer (shared_index_runtime.candidates_
        # persist): L20 writes it, 24/28/32/36 read it inside the captured
        # graph (layer order guarantees write-before-read each step).
        _cbs = int(getattr(indexer, "candidate_block_size", 0) or 0)
        _ctb = int(getattr(indexer, "candidate_topk_blocks", 0) or 0)
        _want_cand = (
            _cbs > 0 and _ctb > 0 and W % _cbs == 0
            and (indexer.is_candidate_source or indexer.uses_candidates))
        if _want_cand:
            from vllm_ascend.models.deepseek_v41.indexer import (
                graph_safe_candidate_blocks as _gcb,
            )

            _cand_pos = _gcb(score, akl, _cbs, _ctb)
            _cbuf = (
                getattr(self.shared_index_runtime, "candidates_persist", None)
                if self.shared_index_runtime is not None else None)
            if _cbuf is None or _cbuf.shape[0] < B or _cbuf.shape[1] < W \
                    or _cbuf.device != dev:
                _cbuf = torch.zeros(max(B, 1), W, dtype=torch.bool, device=dev)
                if self.shared_index_runtime is not None:
                    self.shared_index_runtime.candidates_persist = _cbuf
            if indexer.uses_candidates:
                # non-owns_k: restrict selection to the blocks the source
                # layer published THIS step (same persistent buffer)
                score = score.masked_fill(~_cbuf[:B, :W], -float("inf"))
            else:
                # candidate source: publish for this step's later layers
                _cbuf[:B, :W] = _cand_pos

        if _sel == "sortsel":
            # graph-safe candidate: replace topk (broken in-graph replay) with
            # a full descending argsort + slice + gather; the shared tail
            # re-sorts the selected indices into official position order
            order_full = score.argsort(dim=-1, descending=True)
            idxs = order_full[:, : indexer.index_topk].contiguous()
            vals = score.gather(-1, idxs)
            valid = torch.isfinite(vals)
            topk_tokens = torch.where(
                valid, idxs, torch.full_like(idxs, -1)).to(torch.int32)
            if self.topk_indices_buffer is not None:
                self.topk_indices_buffer[:B].copy_(topk_tokens)
            dbg = getattr(self, "_v41_stepdbg", None)
            if dbg is not None:
                _b = min(B, 8)
                dbg["topk"][:_b] = topk_tokens[:_b, :8].reshape(-1)[:_b]
            return topk_tokens.unsqueeze(1).contiguous()
        if _sel in ("torch", "gsafe"):
            # DEFAULT select: graph_safe_topk — a CANDIDATE exact descending
            # top-k with no sort-family op. NOTE (review fix): this branch
            # MUST run BEFORE any score.topk() call — the earlier revision
            # fell through an unconditional topk first, so gsafe never
            # actually replaced it. "torch.topk in-graph breaks replay" is a
            # HYPOTHESIS pending a properly isolated experiment (the
            # computeonly variant was NOT isolated: it wrote the REAL topk
            # into topk_indices_buffer before returning constants, and
            # non-owns_k layers consume that buffer via
            # _get_shared_topk_indices).
            from vllm_ascend.models.deepseek_v41.indexer import graph_safe_topk

            idxs, _selmask = graph_safe_topk(score, indexer.index_topk)
            vals = score.gather(-1, idxs)
            topk_tokens = torch.where(
                torch.isfinite(vals), idxs,
                torch.full_like(idxs, -1)).to(torch.int32)
            # invalid-index encoding for the graph path: smla's output is
            # bit-identical for -1/0/akl pads (proven on the real captured
            # payload), but -1 consumed by a REPLAYED full-graph smla
            # triggers an aicore trap (cycle51 vs cycle52 A/B) while 0-pad
            # replays clean. Map invalid -> 0 in the graph select output.
            topk_tokens = torch.where(
                topk_tokens < 0, torch.zeros_like(topk_tokens), topk_tokens)
            if self.topk_indices_buffer is not None:
                self.topk_indices_buffer[:B].copy_(topk_tokens)
            dbg = getattr(self, "_v41_stepdbg", None)
            if dbg is not None:
                _b = min(B, 8)
                dbg["topk"][:_b] = topk_tokens[:_b, :8].reshape(-1)[:_b]
            return topk_tokens.unsqueeze(1).contiguous()
        if _sel == "topkconst":
            # bisect: keep k-read + q + score (everything above), replace
            # ONLY the topk/argsort/gather chain with the constant indices
            _ct = getattr(self, "_v41_const_topk_buf", None)
            if _ct is None or _ct.shape[-1] != indexer.index_topk or _ct.device != dev:
                _ct = (torch.arange(indexer.index_topk, device=dev) % 3).to(
                    torch.int32).view(1, 1, -1).contiguous()
                self._v41_const_topk_buf = _ct
            topk_tokens = _ct.view(1, -1).expand(B, -1).contiguous()
            if self.topk_indices_buffer is not None:
                self.topk_indices_buffer[:B].copy_(topk_tokens)
            return topk_tokens.unsqueeze(1).contiguous()
        vals, idxs = score.topk(
            indexer.index_topk, dim=-1,
            sorted=(_sel != "unsorted"),  # bisect: aclnnTopK sorted path may
            # be what breaks in-graph replay; the argsort below restores the
            # official position order either way
        )
        if _sel == "execpure":
            # TRUE compute-only isolation (review fix): run the FULL
            # topk+argsort chain, write its result ONLY to the diagnostic
            # stepdbg tensor (never to topk_indices_buffer — non-owns_k
            # layers consume that buffer and would see the real values),
            # then return the constant index set. Traps here => the sort
            # EXECUTION itself is implicated; clean here => look at the
            # consumed VALUES / buffer path instead.
            valid = torch.isfinite(vals)
            order = idxs.to(torch.float32).argsort(dim=-1)
            idxs = idxs.gather(-1, order)
            valid = valid.gather(-1, order)
            topk_tokens = torch.where(
                valid, idxs, torch.full_like(idxs, -1)).to(torch.int32)
            dbg = getattr(self, "_v41_stepdbg", None)
            if dbg is not None:
                _b = min(B, 8)
                dbg["topk"][:_b] = topk_tokens[:_b, :8].reshape(-1)[:_b]
            _ct = getattr(self, "_v41_const_topk_buf", None)
            if _ct is None or _ct.shape[-1] != indexer.index_topk or _ct.device != dev:
                _ct = (torch.arange(indexer.index_topk, device=dev) % 3).to(
                    torch.int32).view(1, 1, -1).contiguous()
                self._v41_const_topk_buf = _ct
            _out_ct = _ct.expand(B, 1, -1).contiguous()
            # consume EXACTLY what topkconst consumes (const in the shared
            # buffer too) so the only delta vs the CLEAN topkconst run is the
            # topk+argsort EXECUTION itself
            if self.topk_indices_buffer is not None:
                self.topk_indices_buffer[:B].copy_(_out_ct.reshape(B, -1))
            return _out_ct
        if _sel == "noargsort":
            # bisect: keep topk, skip the float32 argsort + gathers (indices
            # unsorted — wrong order, but tests replay of topk alone)
            valid = torch.isfinite(vals)
            topk_tokens = torch.where(
                valid, idxs, torch.full_like(idxs, -1)).to(torch.int32)
            if self.topk_indices_buffer is not None:
                self.topk_indices_buffer[:B].copy_(topk_tokens)
            return topk_tokens.unsqueeze(1).contiguous()
        if _sel == "notopk":
            # bisect: skip topk; constant index set, still run argsort+gather
            _ct2 = getattr(self, "_v41_notopk_buf", None)
            if _ct2 is None or _ct2.shape[-1] != indexer.index_topk:
                _ct2 = (torch.arange(indexer.index_topk, device=dev) % 3).to(
                    torch.float32).view(1, -1).contiguous()
                self._v41_notopk_buf = _ct2
            idxs = _ct2.expand(B, -1).contiguous().long()
            vals = torch.ones_like(idxs, dtype=torch.float32)
        valid = torch.isfinite(vals)
        order = idxs.to(torch.float32).argsort(dim=-1)
        idxs = idxs.gather(-1, order)
        valid = valid.gather(-1, order)
        topk_tokens = torch.where(valid, idxs, torch.full_like(idxs, -1)).to(torch.int32)
        if topk_tokens.shape[-1] < indexer.index_topk:
            pad = torch.full((B, indexer.index_topk - topk_tokens.shape[-1]),
                             -1, dtype=torch.int32, device=dev)
            topk_tokens = torch.cat([topk_tokens, pad], dim=-1)
        if self.topk_indices_buffer is not None:
            self.topk_indices_buffer[:B].copy_(topk_tokens)
        dbg = getattr(self, "_v41_stepdbg", None)
        if dbg is not None:
            _b = min(B, 8)
            dbg["topk"][:_b] = topk_tokens[:_b, :8].reshape(-1)[:_b]
        return topk_tokens.unsqueeze(1).contiguous()

    def _v41_seed_engine_indexer_cache(
        self, layer_name, indexer, indexer_md, kv_cache, keys, req_idx, group_pos
    ) -> None:
        """Prefill seeding of the ENGINE indexer K cache (device-path decode
        reads it; the reference prefill otherwise only fills the dense store).

        Also seeds the deferred-write pending row with the LAST emitted key
        of each request so the first decode step's scatter is real. Runs in
        eager prefill, host-side slot math is fine here.
        """
        from vllm_ascend.device.device_op import DeviceOperator

        dev = keys.device
        (_, ik_cache, isc_cache, ifc_cache) = DeviceOperator.unpack_dsa_indexer_kv_cache(kv_cache)
        rows_per_page = int(ik_cache.shape[1])
        idx_bt = indexer_md.compressor.cache.req_metadata.block_table.to(dev).long()
        gp = group_pos.detach().cpu().tolist()
        ri = req_idx.detach().cpu().tolist()
        # FLAT int32 slots (cache_roundtrip-verified epilog convention)
        flat_list = []
        for r, g in zip(ri, gp):
            col = int(g) // rows_per_page
            page = int(idx_bt[r, col].item())
            flat_list.append(page * rows_per_page + int(g) % rows_per_page)
        slots = torch.tensor(flat_list, dtype=torch.int32, device=dev)
        DeviceOperator.indexer_quant_scatter_part1(
            keys.to(torch.bfloat16), ik_cache,
            ifc_cache if ifc_cache is not None else ik_cache, slots,
        )
        # (deferred-pending machinery removed with the verified same-step
        #  design; seeding only writes the cache rows themselves)

    def _v41_reference_indexer_topk(
        self,
        layer_name: str,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        layer_metadata: AscendDSALayerMetadata,
        compress_kv_cache: torch.Tensor,
        state_cache: torch.Tensor,
        write_cache: bool,
        kv_cache: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        """Dual-run dispatcher (patch 85 VALIDATION mode).

        When V41_DEVICE_INDEXER=1 and this is a true decode step, the device
        path computes the returned top-k while the reference body still runs
        (keeping the reference index-K store and compressor state fresh for
        layers that lack device-side metadata, and for teacher-forced
        comparisons). The device result wins: it is copied over the shared
        topk buffer before returning, so shared consumers see the same
        selection this layer actually used. For graph integration the
        reference run is dropped and every index layer must go device.
        """
        dev_result = None
        _dev_on = __import__("os").environ.get("V41_DEVICE_INDEXER", "0") == "1"
        _dev_handles = (
            getattr(self.shared_index_runtime, "dev_handles", None)
            if self.shared_index_runtime is not None
            else None
        )
        owns_k = bool(getattr(self.indexer, "owns_k", True))
        _req_md = layer_metadata.compressor.cache.req_metadata
        _B = _req_md.num_actual_reqs if _req_md.num_actual_reqs is not None else _req_md.seq_lens.shape[0]
        # ACL graph capture is NOT Dynamo tracing: torch.compiler
        # .is_compiling() stays False while the acl graph records kernels,
        # so detect the capture itself (host syncs are illegal mid-capture).
        _npu_capturing = False
        try:
            _npu_capturing = torch.npu.is_current_stream_capturing()
        except Exception:  # noqa: BLE001
            pass
        # ACL-graph capture ONLY (probe: cap=True/compiling=False during
        # NPU graph capture; compiling=True during prefill Dynamo passes).
        # Replay runs no Python — the captured kernel sequence IS this path
        # — so capture-only gating routes prefill-compile to the eager
        # dispatcher (multi-token batches must seed via the reference impl).
        _graph_decode = _dev_on and _npu_capturing
        if _graph_decode:
            # graph capture/replay: every captured decode step IS one query
            # token per request (the dispatcher's host reads would be frozen
            # anyway); skip the dual reference run — the reference store and
            # compressor state are maintained by prefill forwards.
            return self._v41_device_indexer_topk(
                layer_name=layer_name,
                hidden_states=hidden_states,
                qr=qr,
                req_md=_req_md,
                indexer_md=layer_metadata.indexer,
                kv_cache=kv_cache,
                indexer=self.indexer,
                owns_k=owns_k,
                compress_kv_cache=compress_kv_cache,
            )
        if (
            _dev_on
            and (owns_k or _dev_handles is not None)
            and kv_cache is not None
            and layer_metadata.indexer is not None
            and hidden_states.shape[0] == int(_B)
            and hidden_states.shape[0] > 0
        ):
            _lens = (
                _req_md.query_start_loc[1 : int(_B) + 1] - _req_md.query_start_loc[: int(_B)]
            ).detach().cpu().tolist()
            _sp = _req_md.start_pos[: int(_B)].detach().cpu().tolist()
            if all(int(x) == 1 for x in _lens) and all(int(x) > 0 for x in _sp):
                dev_result = self._v41_device_indexer_topk(
                    layer_name=layer_name,
                    hidden_states=hidden_states,
                    qr=qr,
                    req_md=_req_md,
                    indexer_md=layer_metadata.indexer,
                    kv_cache=kv_cache,
                    indexer=self.indexer,
                    owns_k=owns_k,
                )
        ref_result = self._v41_reference_indexer_topk_impl(
            layer_name=layer_name,
            hidden_states=hidden_states,
            qr=qr,
            layer_metadata=layer_metadata,
            compress_kv_cache=compress_kv_cache,
            state_cache=state_cache,
            write_cache=write_cache,
            kv_cache=kv_cache,
            indexer_md=layer_metadata.indexer,
        )
        if dev_result is not None or ref_result is not None:
            # R21 A-side snapshot: what the indexer bridge DELIVERS here
            try:
                _rm = layer_metadata.compressor.cache.req_metadata
                _bA = int(_rm.num_actual_reqs) if _rm.num_actual_reqs is not None else int(_rm.seq_lens.shape[0])
                _v41_ab_capture(
                    "A", layer_name,
                    dev_result if dev_result is not None else ref_result,
                    _rm.seq_lens[:_bA].detach().cpu().tolist(),
                    _rm.start_pos[:_bA].detach().cpu().tolist(),
                    _rm.block_table[:_bA, 0].detach().cpu().tolist(),
                    _rm.query_start_loc[: _bA + 1].detach().cpu().tolist(),
                )
            except Exception:  # noqa: BLE001
                pass
        if dev_result is not None:
            if __import__("os").environ.get("V41_GSAFE_VERIFY", "0") == "1":
                # eager dual-run cross-check: graph_safe_topk vs the reference
                # torch.topk selection (tie-at-boundary divergence would show
                # here; expected 0 mismatches — gsafe_topk_test.py proved set
                # equality on all non-adversarial cases)
                _d = dev_result.reshape(dev_result.shape[0], -1)
                _r = ref_result.reshape(ref_result.shape[0], -1)
                if _d.shape == _r.shape and not torch.equal(_d, _r):
                    _n = int((_d != _r).sum())
                    print(f"[gsafe-verify] {layer_name} mismatch {_n}/{_d.numel()} "
                          f"first dev={_d[_d != _r][:4].tolist()} ref={_r[_d != _r][:4].tolist()}",
                          flush=True)
            if self.topk_indices_buffer is not None:
                self.topk_indices_buffer[: dev_result.shape[0]].copy_(
                    dev_result.reshape(dev_result.shape[0], -1)
                )
            return dev_result
        return ref_result

    def _v41_reference_indexer_topk_impl(
        self,
        layer_name: str,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        layer_metadata: AscendDSALayerMetadata,
        compress_kv_cache: torch.Tensor,
        state_cache: torch.Tensor,
        write_cache: bool,
        kv_cache: tuple[torch.Tensor, ...] | None = None,
        indexer_md=None,
    ) -> torch.Tensor:
        """V4.1 bridge: drive the reference indexer (deepseek_v41/indexer.py).

        owns_k layers (kv_source 2/8/14/20) pool compressor latents with the
        official math, rotate and append their index keys into the indexer's
        dense reference K store, then select top-k against it; non-owns_k
        index sources (24/28/32/36) score the store the most recent owner
        published (official SharedAttentionRuntime.index_k) and mask by the
        published candidates. The fused compressor op still runs first for
        kv_source layers so the compressed KV cache the attention reads is
        written. Official top-k semantics with offset=0: the paged sparse
        attention op indexes the compressed cache directly, so no window-KV
        offset is applied.
        """
        indexer = self.indexer
        req_md = layer_metadata.compressor.cache.req_metadata
        assert req_md is not None, "V4.1 indexer bridge needs request metadata"
        owns_k = bool(getattr(indexer, "owns_k", True))

        num_tokens = hidden_states.shape[0]
        device = hidden_states.device
        ratio = indexer.compress_ratio
        B_active = int(req_md.num_actual_reqs) if req_md.num_actual_reqs is not None else req_md.seq_lens.shape[0]
        qsl = req_md.query_start_loc
        seq_lens = req_md.seq_lens
        start_pos = req_md.start_pos
        assert start_pos is not None, "V4.1 indexer bridge needs per-request start_pos"

        # One host sync for per-request identity/lengths (reference path).
        # Request identity is the first KV block id (block_table[i, 0]),
        # stable for a request's lifetime.
        # Plain-memcpy D2H then host-side widening: a fused cast+copy
        # dispatches to the AICPU RunCp kernel, which rejects these dtypes.
        qsl_cpu = qsl[: B_active + 1].detach().cpu().to(torch.int64).tolist()
        seq_lens_cpu = seq_lens[:B_active].detach().cpu().to(torch.int64).tolist()
        first_blocks = req_md.block_table[: B_active, 0].detach().cpu().to(torch.int64).tolist()
        assert qsl_cpu[B_active] == num_tokens, "query_start_loc disagrees with the token count"
        if __import__("os").environ.get("V41_TEAR_GUARD", "1") != "0":
            # R39: this device readback tore under eval load (the R38b fatal
            # crash: max compress width became 3 for a 439-token chunk —
            # seq_lens read back as 6/7). R39d: trigger ONLY on the
            # impossible-by-construction signature (a request's total length
            # below its OWN current chunk, or beyond max_model_len) — the
            # R39/R39b store-reconciliation invariant misfired on
            # prefix-cached continuations (rows cover the full cached
            # history) and mass-misrepaired healthy eval batches.
            _chunks = [qsl_cpu[b + 1] - qsl_cpu[b] for b in range(B_active)]
            _repaired = 0
            for _b in range(B_active):
                _sl = seq_lens_cpu[_b]
                if _chunks[_b] <= _sl <= 65536:
                    continue
                _rows = int(indexer.ref_slot_len.get(first_blocks[_b], 0))
                _cand = (_rows * ratio + _chunks[_b] if owns_k
                         else _rows * ratio) if _rows > 0 else _chunks[_b]
                seq_lens_cpu[_b] = max(min(_cand, 65536), _chunks[_b])
                _repaired += 1
            if _repaired:
                print(f"[V41_LEN_GUARD] pid={__import__('os').getpid()} "
                      f"repaired={_repaired}", flush=True)
        # R41: the width must NEVER come from a device readback — two
        # independent tears now proved it fatal (R38b: seq_lens read 6/7
        # -> width 3; r40: qsl readback torn so the chunk reference itself
        # lied and the guard passed a width of 4 against q's S=473).
        # Derive compress widths from the K store's HOST-side row counts
        # (pure host facts, immune) with a per-request chunk floor; extra
        # width rows are zero-padded and masked by cl, so over-estimating
        # is safe while under-estimating traps the einsum/op.
        _chunks_h = [qsl_cpu[b + 1] - qsl_cpu[b] for b in range(B_active)]
        compress_lens = [
            max(sl // ratio,
                (_chunks_h[b] + ratio - 1) // ratio if _chunks_h[b] >= 8 else 0)
            for b, sl in enumerate(seq_lens_cpu)
        ]
        max_cmp = max(compress_lens, default=0)

        # Indexer queries rotate at the QUERY token positions with this
        # layer's compressed rope (compress theta): complex lanes (every
        # other column of the duplicated cos/sin tables), [T, 1, rd // 2].
        # Per-token absolute positions (start_pos[i] + arange(len_i)),
        # matching the official freqs_cis[start_pos:end_pos] slicing.
        # R23 race fix: derive every per-request scalar from the host-synced
        # qsl/seq_lens lists and upload FRESH tensors. Reading the device
        # start_pos/seq_lens/qsl views here can tear under async scheduling
        # (fault caps: cl=[1]*16+[17..] with start_pos=[0,0]; ratio-2 dump
        # showed a +16 group-position shift = torn start_pos read by the
        # pool). Fresh uploads are produced on this stream AFTER the host
        # sync, so they cannot carry another step's partial content.
        # R31 (r24fix edit 1 ported): compute q_pos from HOST batch facts —
        # the engine's device start_pos/seq_lens/qsl views can be torn by
        # the mid-forward writer; host-built ranges uploaded after the host
        # sync cannot carry another step's partial content.
        _sp_cpu = [
            int(seq_lens_cpu[b]) - (qsl_cpu[b + 1] - qsl_cpu[b])
            for b in range(B_active)
        ]
        q_pos = torch.cat([
            torch.arange(sp, sp + (q1 - q0), dtype=torch.int64)
            for q0, q1, sp in zip(qsl_cpu[:-1], qsl_cpu[1:], _sp_cpu)
        ]).to(device)
        query = indexer.compute_query(qr, q_pos)

        _dbg_idx_on = _v41_dbg_enabled("idx")
        _dbg_appended = None
        _dbg_index_k = None
        _dbg_score_pre = None
        _dbg_sp_in = None
        _dbg_arange = None
        _dbg_cl_just = None
        _dbg_newcand_vals = None
        _dbg_newcand_id = None
        _cand_src = None
        _dbg_cl_guard_dev = None
        _dbg_cl_raw = None

        if owns_k:
            # Always pool, even when nothing is visible yet: official decode
            # writes the state slot every step so a group completing later
            # pools over all its tokens.
            compressor = self.compressor
            assert compressor is not None and indexer.ref_pool is not None
            wgate_weight = compressor.wgate.weight if compressor.wgate is not None else None
            # R31 (r24fix edit 2 ported): pool reads host-built qsl/sl/sp
            # uploads, not the engine device views (tear-immune source).
            _qsl_dev = torch.tensor(qsl_cpu[: B_active + 1], dtype=seq_lens.dtype, device=device)
            _sl_dev = torch.tensor(seq_lens_cpu, dtype=seq_lens.dtype, device=device)
            _sp_dev = torch.tensor(_sp_cpu, dtype=start_pos.dtype, device=device)
            latents, req_idx, group_pos = indexer.ref_pool.pool(
                hidden_states,
                compressor.wkv.weight,
                wgate_weight,
                compressor.norm.weight,
                _qsl_dev,
                _sl_dev,
                _sp_dev,
                req_keys=first_blocks,
            )
            # prefill -> graph-decode hand-off: the trailing PARTIAL group's
            # kv/score carry must live in the device pool rows before the
            # first decode step completes that group (engine-slot identity)
            self._v41_seed_device_pool_carry(
                layer_name, indexer, req_md, B_active, first_blocks, seq_lens_cpu)
            if latents.shape[0] > 0:
                # Key rope: the full_compress tables are token-position
                # tables and a latent for compressed position g rotates at
                # the group-start token position g * ratio (official
                # semantics; the fused compressor op applies the same).
                fc = req_md.full_compress_cos
                fs = req_md.full_compress_sin
                assert fc is not None and fs is not None
                # A latent for compressed position g stands for the group's
                # FIRST token: official rotates it at token position g*ratio.
                token_pos = (group_pos.reshape(-1) * ratio).to(device=device, dtype=torch.int64)
                keys = indexer.compute_index_key(latents, token_pos)
                if (
                    __import__("os").environ.get("V41_DEVICE_INDEXER", "0") == "1"
                    and __import__("os").environ.get("V41_DEV_SKIP_SCATTER", "0") != "1"
                    and kv_cache is not None
                    and indexer_md is not None
                ):
                    self._v41_seed_engine_indexer_cache(
                        layer_name, indexer, indexer_md, kv_cache,
                        keys, req_idx, group_pos,
                    )
                req_slots = torch.tensor(first_blocks, dtype=torch.int64, device=device)[req_idx]
                if __import__("os").environ.get("V41_SKIP_REF_APPEND", "0") != "1":
                    indexer.append_ref_keys(keys, req_slots, group_pos)
                    if _dbg_idx_on:
                        _dbg_appended = (
                            group_pos.reshape(-1).detach().cpu(),
                            keys.detach().cpu(),
                            int(first_blocks[0]) if first_blocks else -1,
                            dict(indexer.ref_slot_of_req),
                            dict(indexer.ref_slot_len),
                            int(indexer.ref_total),
                        )

                if write_cache and __import__("os").environ.get("V41_SKIP_CMP_WRITE", "0") != "1":
                    # Compressed KV write: shared cmp_kv_from_latent helper
                    # (normed latent -> rope at g*ratio -> bf16 -> fp4-e4m3scale
                    # per 16) — the SAME function the graph decode scatter
                    # uses, so the two paths cannot drift.
                    #
                    # Slot geometry: the compressed pool allocates pages of
                    # ``compress_kv_cache.shape[1]`` rows, ONE ROW PER
                    # COMPRESSED POSITION (capture-verified: 252 ratio-2
                    # positions -> 2 pages of 128 rows; 505 ratio-1 positions
                    # -> 4 pages). So compressed position g lives at
                    # (cmp_block_table[g // rows_per_page], g % rows_per_page).
                    from vllm_ascend.models.deepseek_v41.indexer import cmp_kv_from_latent

                    cmp_kv = cmp_kv_from_latent(indexer, latents, token_pos)
                    cmp_pos = group_pos.reshape(-1).to(device=device, dtype=torch.int64)
                    rows_per_page = int(compress_kv_cache.shape[1])
                    bt = req_md.block_table
                    col = cmp_pos // rows_per_page
                    blk = bt.index_select(0, req_idx.to(device=device, dtype=torch.int64)) \
                        .gather(1, col.clamp(max=bt.shape[1] - 1).unsqueeze(1)).squeeze(1)
                    off = cmp_pos % rows_per_page
                    slots = torch.stack([blk, off], dim=-1).to(torch.int32)
                    get_dsa_attn_kv_plan(self.vllm_config).dsa_kv_compress_scatter(
                        compress_kv_cache,
                        cmp_kv.view(cmp_kv.shape[0], 1, cmp_kv.shape[-1]),
                        slots,
                    )
            if self.shared_index_runtime is not None:
                # publish the provider for the non-owns_k index sources above
                self.shared_index_runtime.index_k_provider = indexer.stack_ref_keys

        if max_cmp == 0:
            # no visible compressed position yet (official returns an empty
            # top-k without running the indexer): all slots unreachable
            topk_tokens = torch.full(
                (num_tokens, indexer.index_topk), -1, dtype=torch.int32, device=device
            )
        else:
            live_rows = None
            if owns_k:
                index_k, live_rows = indexer.stack_ref_keys(first_blocks, compress_lens, layer=layer_name)
            else:
                runtime = self.shared_index_runtime
                provider = runtime.index_k_provider if runtime is not None else None
                if provider is None:
                    raise RuntimeError(
                        "V4.1 non-owns_k indexer ran before an owns_k layer published "
                        "the shared index-K provider"
                    )
                index_k, prov_live = provider(first_blocks, compress_lens, layer=layer_name)
                if live_rows is None:
                    live_rows = prov_live
            if _dbg_idx_on:
                _dbg_index_k = index_k.detach().cpu()

            # per-query visible compressed positions [B, S, 1]: a block becomes
            # visible once the query has passed its last token
            S = max(q1 - q0 for q0, q1 in zip(qsl_cpu[:-1], qsl_cpu[1:]))
            _ar = torch.arange(S, device=device)
            if _dbg_idx_on:
                # R27 points 1+2: snapshot the ENGINE's actual input views
                # BEFORE cl is minted (value snapshots; observability of the
                # raw engine tensors — the fix below no longer consumes them).
                _dbg_sp_in = start_pos[:B_active].detach().clone()
                _dbg_arange = _ar.detach().clone()
            # R31 (r24fix edit 3 ported): cl values computed on HOST from the
            # batch facts synced at the top of this call; uploaded fresh. A
            # torn device start_pos/arange chain can no longer mint a wrong
            # cl — the observed fault (M1 correct -> M2 [1,1,1,...]) drew
            # from device-resident values.
            cl_host = torch.tensor(
                [[_sp_cpu[b] + t + 1 for t in range(S)] for b in range(B_active)],
                dtype=torch.int64,
            ) // ratio
            # R32 prefix guard: the observed writer stamps a run of int64
            # ONES at an allocation's HEAD (16 slots = 128 bytes on cl; the
            # pool's {0,8,16,24,32} is the same stamp read as bool). Offset
            # the live data 16 slots into its storage: a head stamp lands in
            # the sentinel prefix (-7 / -9) and the consumed view stays
            # intact; the prefix also acts as a strike counter for the caps.
            _GUARD = 16
            cl_full = torch.full(
                (_GUARD + B_active * S,), -7, dtype=torch.int64)
            cl_full[_GUARD:] = cl_host.reshape(-1)
            cl_dev = cl_full.to(device=device, non_blocking=False)
            cl = cl_dev[_GUARD:].view(B_active, S, 1)
            if _dbg_idx_on:
                # R26 point 3 (kept): cl JUST minted (now the upload).
                _dbg_cl_just = cl.detach().clone()
            # R31/R32/R35 tear guard: second upload with its own sentinel
            # prefix, consumed inside score() right before the visibility
            # masked_fill (after the scatter-block window where cl was
            # observed torn); elementwise where-recover, no host sync.
            # R35: the backup's DATA sits at a staggered offset (+80 extra
            # slots = +640B) — the R34 eval crash showed a single-slot
            # INTERIOR tear; two identically-laid-out adjacent copies get
            # stamped at the same logical slots and become indistinguishable,
            # while staggered copies keep the where decisive.
            _tg = __import__("os").environ.get("V41_TEAR_GUARD", "1") != "0"
            cl_backup = None
            if _tg:
                _BKOFF = _GUARD + 80
                bk_full = torch.full(
                    (_BKOFF + B_active * S,), -9, dtype=torch.int64)
                bk_full[_BKOFF:] = cl_host.reshape(-1)
                cl_backup = bk_full.to(
                    device=device, non_blocking=False)[_BKOFF:].view(
                        B_active, S, 1)
            _dbg_cl_guard_dev = cl_dev[:_GUARD]
            # R42: single source of truth for query_start_loc — upload the
            # host-synced qsl_cpu instead of passing the engine's device
            # view. Three fatal width crashes (3/4/5 vs 439/473/264) all
            # came from score()'s S (device qsl) disagreeing with the
            # bridge's host-derived width; one uploaded source makes the
            # einsum mismatch structurally impossible.
            _qsl_t = torch.tensor(qsl_cpu, dtype=torch.int32,
                                  device=device)
            # R43 pre-score forensic gate: the einsum demands width*S
            # consistency with the same host list; if the actual stacked
            # width disagrees with S, dump EVERY host fact to a file and
            # fail loudly BEFORE the op — the width crashes (3/4/5/2 vs
            # 439/473/264/394) all died inside score without a trace.
            # R51: a LEGAL pool holds floor(S/ratio) whole groups — the
            # trailing partial group is not yet compressed (7 tokens at
            # ratio 2 -> width 3, 3*2=6<7). The original form (< S) fired
            # on exactly that legal short-prompt case and killed the engine
            # on its first 7-token sanity request. Real tears are deficits
            # of WHOLE groups (width 3 vs S=439), so gate on that bound.
            _S_h = max(qsl_cpu[b + 1] - qsl_cpu[b] for b in range(B_active)) \
                if B_active else 0
            if index_k.shape[1] < _S_h // ratio:
                import json as _json
                import os as _os_f
                _rec = {
                    "pid": _os_f.getpid(), "layer": layer_name,
                    "qsl_cpu": qsl_cpu, "seq_lens_cpu": seq_lens_cpu,
                    "chunks": _chunks_h, "compress_lens": compress_lens,
                    "max_cmp": max_cmp, "S": _S_h,
                    "index_k_width": int(index_k.shape[1]),
                    "ratio": ratio, "num_tokens": int(num_tokens),
                    "B_active": int(B_active),
                    "first_blocks": list(first_blocks),
                    "sl_readback_raw": seq_lens[:B_active].detach().cpu()
                                       .to(torch.int64).tolist(),
                    "qsl_readback_raw": qsl[: B_active + 1].detach().cpu()
                                        .to(torch.int64).tolist(),
                }
                _f = f"/ws/evidence/p85run2/r38/width_crash_{_os_f.getpid()}.json"
                with open(_f, "w") as _fh:
                    _json.dump(_rec, _fh, indent=1)
                raise RuntimeError(
                    f"R43 width gate: index_k width {index_k.shape[1]} "
                    f"cannot serve S={_S_h} (facts dumped to {_f})")
            index_score = indexer.score(
                hidden_states, query, index_k, cl,
                query_start_loc=_qsl_t, cl_backup=cl_backup,
            )
            if cl_backup is not None:
                # R33/R36: score() heals only its LOCAL compress_lens and
                # returns just the scores. R33 healed via torch.where, but
                # the where OUTPUT is an unguarded fresh allocation — the
                # R35 eval crash showed a tear landing exactly there (3
                # slots read 91/91/87 at select_topk while score_pre had
                # 321 finite). R36: downstream consumes the backup DIRECTLY
                # (guarded prefix + staggered offset + host-minted values);
                # no unguarded heal temp exists on this path.
                if _dbg_idx_on:
                    _dbg_cl_raw = cl.detach().clone()
                cl = cl_backup
            if _dbg_idx_on:
                # R23: raw scores BEFORE any post-score filter — the
                # per-position comparison table needs the unfiltered values
                _dbg_score_pre = index_score.detach().cpu()
            if live_rows is not None:
                # Mask rows past a request's live length ALWAYS (the previous
                # gate compared live rows against the query count cl.shape[1],
                # which is 1 on decode, so decode-side missing rows were never
                # masked). With the strict store (patch 82) live_rows equals
                # compress_lens or the read already raised; this remains a
                # belt for requests with compress_len == 0 in mixed batches.
                pos = torch.arange(index_score.shape[-1], device=index_score.device)
                live_t = torch.tensor(live_rows, device=index_score.device, dtype=torch.float32).view(-1, 1, 1)
                index_score = index_score.masked_fill(pos.view(1, 1, -1) >= live_t, -torch.inf)

            candidates = None
            _cand_src = None
            if not owns_k:
                candidates = self.candidates_buffer
                if candidates is not None:
                    _cand_src = "local"
                elif self.shared_index_runtime is not None:
                    candidates = self.shared_index_runtime.candidates
                    _cand_src = "runtime"
                if _tg:
                    # R31/R32 tear guard: consume a FRESH upload of the host
                    # copy published by the source layer, with a 64-byte
                    # sentinel prefix (the pool stamp observed = 40 bytes of
                    # ONES at the storage head). A device-side tear between
                    # publish and this read cannot reach the consumed
                    # values; a head stamp lands in the False prefix. The
                    # shared device buffer stays published for the B
                    # watchpoints (observability).
                    _ch = getattr(self, "candidates_host", None)
                    if _ch is None and self.shared_index_runtime is not None:
                        _ch = getattr(self.shared_index_runtime,
                                      "candidates_host", None)
                    if _ch is not None:
                        _g = 64
                        _pool_full = torch.zeros(
                            _g + _ch.numel(), dtype=torch.bool)
                        _pool_full[_g:] = _ch.reshape(-1)
                        candidates = _pool_full.to(device)[_g:].view(
                            _ch.shape[0], _ch.shape[1], _ch.shape[2])
                        _cand_src = "guard_upload"
                if candidates is not None and candidates.shape != index_score.shape:
                    # defensive realignment; same step/requests should match
                    width = index_score.shape[-1]
                    cand = candidates[..., :width]
                    if cand.shape[-1] < width:
                        pad = torch.zeros(
                            (*cand.shape[:-1], width - cand.shape[-1]),
                            dtype=torch.bool,
                            device=cand.device,
                        )
                        cand = torch.cat([cand, pad], dim=-1)
                        _cand_src = "realign_cat"
                    else:
                        _cand_src = "realign_slice"
                    candidates = cand

            topk = indexer.select_topk(
                index_score,
                cl,
                end_pos=max(seq_lens_cpu),
                offset=0,
                candidates=candidates,
            )
            if _tg and topk.dim() == 3:
                # R52-AB: same-boot A/B switch — writing "old" into the
                # mode file restores the R39c value-exact guard (the L24
                # bug) for this call; anything else runs the R52 structural
                # guard. Read per call so one boot serves both A/B sides.
                _gm_old = False
                try:
                    with open("/ws/evidence/p85run2/r52ab/guard_mode") as _gmf:
                        _gm_old = _gmf.read().strip() == "old"
                except OSError:
                    pass
                if _gm_old:
                    # ---- R39c legacy path, kept verbatim for the A/B ----
                    _lens_h = torch.tensor(
                        [qsl_cpu[b + 1] - qsl_cpu[b]
                         for b in range(B_active)], dtype=torch.int64)
                    _real_t = torch.arange(S).view(1, -1) < \
                        _lens_h.view(-1, 1)
                    _exp = cl_host.clamp(max=min(topk.shape[-1], max_cmp))
                    _ar3 = torch.arange(topk.shape[-1]).view(1, 1, -1)
                    _expected = torch.where(
                        _ar3 < _exp.view(B_active, S, 1), _ar3,
                        torch.full_like(_ar3, -1)).to(topk.dtype)
                    _t_cpu = topk.cpu()
                    _bad = ((_t_cpu != _expected) &
                            _real_t.unsqueeze(-1)).any(-1) \
                        .nonzero(as_tuple=False)
                    if _bad.numel():
                        import os as _os_g
                        print(f"[V41_SEL_GUARD-OLD] pid={_os_g.getpid()} "
                              f"total={max(seq_lens_cpu)} "
                              f"rows={int(_bad.shape[0])} "
                              f"first={_bad[0].tolist()}", flush=True)
                        _t_cpu[_bad[:, 0], _bad[:, 1]] = _expected[
                            _bad[:, 0], _bad[:, 1]]
                        topk = _t_cpu.to(device)
                else:
                    # R52 selection guard. R38/R39c verified VALUES against the
                    # exact expectation [0..cl-1] — valid only when k >= cl
                    # (select-all). For cl > k the OFFICIAL selection is top-k
                    # BY SCORE, scattered in position space (frozen 55f597c
                    # Indexer.forward); the value-exact form rewrote every
                    # healthy long-context row to [0..k-1] (L24 t1424 cap:
                    # 267 of 512 delivered positions scored strictly below the
                    # official cutoff, 0/512 overlap with the official set).
                    # Structural validity instead (offline harness r52_harness2):
                    #   deficit: valid count < target = min(k, |cand ∩ visible|)
                    #   duplicates / unsorted / any value >= cl
                    # Repair: official recompute from the effective scores when
                    # they are healthy (finite >= target); host rebuild
                    # [0..min(cl,k)) only when the scores themselves are torn
                    # (R32 fault: exact whenever cl <= k).
                    # Only REAL token rows verified (R38b: padding rows
                    # legitimately select nothing).
                    _kw = topk.shape[-1]
                    _t_cpu = topk.cpu().long()
                    _valid = _t_cpu >= 0
                    _cnt = _valid.sum(-1)
                    if candidates is not None:
                        _ar_w = torch.arange(
                            candidates.shape[-1],
                            device=candidates.device).view(1, 1, -1)
                        _vis = _ar_w < cl.reshape(B_active, S, 1).to(
                            candidates.device)
                        _tgt = (candidates & _vis).sum(-1) \
                            .clamp(max=_kw).cpu()
                    else:
                        _tgt = cl_host.clamp(min=0, max=_kw)
                    _bad = _cnt < _tgt
                    _t2 = torch.where(
                        _valid, _t_cpu.float(),
                        torch.full_like(_t_cpu.float(), float("inf")))
                    _nxt, _prv = _t2[:, :, 1:], _t2[:, :, :-1]
                    _asc = ((_nxt > _prv) |
                            (_nxt == float("inf"))).all(-1)
                    _bad = _bad | (~_asc)
                    _inr = ((_t_cpu < cl_host.view(B_active, S, 1)) |
                            ~_valid).all(-1)
                    _bad = _bad | (~_inr)
                    _real_t = torch.arange(S).view(1, -1) < torch.tensor(
                        [qsl_cpu[b + 1] - qsl_cpu[b]
                         for b in range(B_active)]).view(-1, 1)
                    _bad = (_bad & _real_t).nonzero(as_tuple=False)
                    if _bad.numel():
                        import os as _os_g
                        # R53: violation breakdown + repair modes — answers
                        # "is the writer still alive, where, what signature"
                        # from the live service without tensor capture.
                        _v_def = int((_cnt[_bad[:, 0], _bad[:, 1]] <
                                      _tgt[_bad[:, 0], _bad[:, 1]]).sum())
                        _v_ord = int((~_asc[_bad[:, 0], _bad[:, 1]]).sum())
                        _v_rng = int((~_inr[_bad[:, 0], _bad[:, 1]]).sum())
                        _eff = torch.nan_to_num(
                            index_score.masked_fill(
                                ~candidates, -float("inf"))
                            if candidates is not None else index_score,
                            nan=-float("inf"))
                        _n_official = 0
                        _n_host = 0
                        for _i in range(_bad.shape[0]):
                            _bi, _si = int(_bad[_i, 0]), int(_bad[_i, 1])
                            _row = _eff[_bi, _si]
                            _nfin = int(torch.isfinite(_row).sum())
                            if _nfin >= int(_tgt[_bi, _si]) and \
                                    int(_tgt[_bi, _si]) > 0:
                                _n_official += 1
                                # official tail, verbatim (55f597c): top-k by
                                # score, position sort, visibility filter
                                _ids = _row.topk(
                                    min(_kw, _row.shape[-1]), dim=-1,
                                    sorted=False).indices.sort(
                                        dim=-1).values
                                _ids = _ids[_ids < int(cl_host[_bi, _si])]
                                _new = torch.full(
                                    (_kw,), -1, dtype=torch.long,
                                    device=_row.device)
                                _new[:_ids.numel()] = _ids
                                _t_cpu[_bi, _si] = _new.cpu()
                            else:
                                _n_host += 1
                                # scores torn: host rebuild (exact for cl<=k)
                                _n = min(int(cl_host[_bi, _si]), _kw)
                                _new = torch.full((_kw,), -1,
                                                  dtype=torch.long)
                                _new[:_n] = torch.arange(_n)
                                _t_cpu[_bi, _si] = _new
                        print(f"[V41_SEL_GUARD] pid={_os_g.getpid()} "
                              f"L={layer_name} "
                              f"total={max(seq_lens_cpu)} "
                              f"rows={int(_bad.shape[0])} "
                              f"deficit={_v_def} order={_v_ord} "
                              f"range={_v_rng} "
                              f"repair=official:{_n_official}+host:{_n_host} "
                              f"first={_bad[0].tolist()}", flush=True)
                        topk = _t_cpu.to(device=device).to(topk.dtype)
            # unpad request-major rows back to token order
            # R46: replace repeat_interleave + advanced-index gather with
            # per-request slice copies — same aicore fingerprint as score()
            _rows = []
            for _b in range(B_active):
                _st = qsl_cpu[_b]
                _en = qsl_cpu[_b + 1]
                if _en > _st:
                    _rows.append(topk[_b, : _en - _st])
            topk_tokens = torch.cat(_rows, dim=0) if _rows else \
                topk.new_zeros(0, topk.shape[-1])
            # pad to the fixed index_topk width with -1 (unreachable)
            if topk_tokens.shape[-1] < indexer.index_topk:
                pad = torch.full(
                    (num_tokens, indexer.index_topk - topk_tokens.shape[-1]),
                    -1,
                    dtype=topk_tokens.dtype,
                    device=device,
                )
                topk_tokens = torch.cat([topk_tokens, pad], dim=-1)

            if indexer.is_candidate_source:
                new_candidates = indexer.compute_candidates(index_score, cl)
                self.candidates_buffer = new_candidates
                if self.shared_index_runtime is not None:
                    self.shared_index_runtime.candidates = new_candidates
                if _tg:
                    # R31 tear guard: host copy of the published pool —
                    # readers consume fresh uploads of this (see above).
                    _cand_host = new_candidates.detach().cpu()
                    self.candidates_host = _cand_host
                    if self.shared_index_runtime is not None:
                        self.shared_index_runtime.candidates_host = _cand_host
                if _dbg_idx_on:
                    # R28 write-side evidence: identity + WRITE-TIME value
                    # snapshot of the pool this layer just published (device
                    # clone; read back once at save time — no mid-forward
                    # host sync).
                    _dbg_newcand_vals = new_candidates.detach().clone()
                    _dbg_newcand_id = {
                        "storage_ptr": new_candidates.storage().data_ptr(),
                        "data_ptr": new_candidates.data_ptr(),
                        "offset": new_candidates.storage_offset(),
                        "stride": list(new_candidates.stride()),
                        "shape": list(new_candidates.shape),
                    }

        # publish for the compressed layers that reuse this top-k
        if self.topk_indices_buffer is not None:
            self.topk_indices_buffer[:num_tokens].copy_(topk_tokens)

        if _dbg_idx_on:
            try:
                _total = max(seq_lens_cpu)
                if num_tokens >= 32:
                    import os as _os_p
                    print(f"[V41_DBGPROBE] pid={_os_p.getpid()} L={layer_name.split('.')[2] if '.' in layer_name else layer_name} "
                          f"total={_total} total_ok={_v41_dbg_total_ok(_total)} "
                          f"layer_ok={_v41_dbg_layer_ok(layer_name, 'V41_CAPTURE_IDX_LAYERS')} "
                          f"n_before={_V41_DBG.get('n', {}).get((_os_p.getpid(), ( __import__('re').search(r'layers.(\d+).', layer_name) or [None,'x'] )[1], 'idx'), 0)}",
                          flush=True)
                if _v41_dbg_total_ok(_total) and _v41_dbg_layer_ok(layer_name, "V41_CAPTURE_IDX_LAYERS"):
                    _v41_dbg_save(
                        {
                            "owns_k": owns_k,
                            "ratio": ratio,
                            "num_tokens": num_tokens,
                            "seq_lens": seq_lens_cpu,
                            "start_pos": start_pos[:B_active].detach().cpu().tolist(),
                            "is_candidate_source": indexer.is_candidate_source,
                            "uses_candidates": indexer.uses_candidates,
                            "topk_tokens": topk_tokens.detach().cpu(),
                            "index_k": _dbg_index_k,
                            "appended": _dbg_appended,
                            "region_meta": (
                                dict(indexer.ref_slot_of_req), dict(indexer.ref_slot_len), int(indexer.ref_total)
                            )
                            if owns_k
                            else None,
                            # R21: the three -1 sources' actual values
                            "cl": cl.detach().cpu(),
                            # R23: scores pre-filter and post-live-mask
                            "score_pre": _dbg_score_pre,
                            # R26 four points (cloned value snapshots)
                            "sp_in": _dbg_sp_in.cpu() if _dbg_sp_in is not None else None,
                            "arange": _dbg_arange.cpu() if _dbg_arange is not None else None,
                            "cl_just": _dbg_cl_just.cpu() if _dbg_cl_just is not None else None,
                            "cl_used": getattr(indexer, "debug_cl_used", None).cpu()
                            if getattr(indexer, "debug_cl_used", None) is not None else None,
                            # R29 micro-brackets: compress_lens snapshotted
                            # after each op group inside score() (M0 entry,
                            # M1 weights_proj, M2 scatter, M3 einsum+relu_,
                            # M4 weighted-sum) + ptr on both sides — proves
                            # same allocation and names the tearing window.
                            "cl_marks": [m.cpu() for m in
                                         (getattr(indexer, "debug_cl_marks", None) or [])],
                            "cl_ptr": {
                                "bridge": cl.data_ptr(),
                                "score": getattr(indexer, "debug_cl_ptr", None),
                            },
                            "cl_recovered": getattr(
                                indexer, "debug_cl_recovered", None),
                            # R32: the sentinel prefix at save time — still
                            # -7 means no strike; ONES here mean the writer
                            # stamped the storage head and the guard
                            # absorbed it.
                            "cl_guard": _dbg_cl_guard_dev.cpu().tolist()
                            if _dbg_cl_guard_dev is not None else None,
                            "cl_raw": _dbg_cl_raw.cpu()
                            if _dbg_cl_raw is not None else None,
                            # R25: RAW scores (post weighted-sum, pre any
                            # masked_fill) stashed inside indexer.score,
                            # plus explicit same-call/request identity
                            "score_raw": getattr(indexer, "debug_raw_score", None).detach().cpu()
                            if getattr(indexer, "debug_raw_score", None) is not None else None,
                            "first_blocks": list(first_blocks),
                            "qsl": list(qsl_cpu),
                            "call_no": _V41_DBG.get("n", {}).get(
                                (__import__("os").getpid(),
                                 (__import__("re").search(r"layers\.(\d+)\.", layer_name) or [None, "x"])[1],
                                 "idx"), 0),
                            "score_post": index_score.detach().cpu(),
                            # R22: per-row NaN census of the scores AND the
                            # stacked K view rows — discriminates "K store
                            # rows corrupted between L20 and L24 reads"
                            # (view rows NaN) from score-path issues.
                            "index_k_row_nan": (
                                torch.isnan(_dbg_index_k.float())
                                .any(dim=-1).sum(dim=-1).tolist()
                                if _dbg_index_k is not None else None),
                            "index_k_row_absmax": (
                                _dbg_index_k.float().abs()
                                .amax(dim=-1).tolist()
                                if _dbg_index_k is not None else None),
                            "live_rows": list(live_rows) if live_rows is not None else None,
                            "candidates_none": candidates is None,
                            "candidates_shape": tuple(candidates.shape) if candidates is not None else None,
                            "candidates_values": candidates.detach().cpu() if candidates is not None and candidates.numel() <= (1 << 22) else None,
                            # R28 read-side evidence: which buffer this layer's
                            # select actually consumed + its storage identity
                            "cand_id": (
                                {"src": _cand_src,
                                 "storage_ptr": candidates.storage().data_ptr(),
                                 "data_ptr": candidates.data_ptr(),
                                 "offset": candidates.storage_offset(),
                                 "stride": list(candidates.stride()),
                                 "shape": list(candidates.shape),
                                 "used": bool(indexer.uses_candidates)}
                                if candidates is not None else None),
                            "newcand_vals": _dbg_newcand_vals.cpu()
                            if _dbg_newcand_vals is not None else None,
                            "newcand_id": _dbg_newcand_id,
                            "ref_written": dict(indexer.ref_written) if owns_k else None,
                        },
                        _v41_dbg_path("V41_CAPTURE_IDX", layer_name, _total, "idx"),
                    )
            except Exception as _e:  # noqa: BLE001 - debug hook must never kill serving
                import traceback as _tb
                print(f"[V41_DBG_SAVE_ERR] {type(_e).__name__}: {_e}\n"
                      + "\n".join(_tb.format_exc().splitlines()[-4:]),
                      flush=True)

        # [T, 1, index_topk] int32 for cmp_sparse_indices; -1 marks unreachable
        return topk_tokens.unsqueeze(1).contiguous()

    def _get_shared_topk_indices(self, num_tokens: int) -> torch.Tensor:
        if self.topk_indices_buffer is None:
            raise RuntimeError(
                "topk_indices_buffer is required to read the shared indexer top-k"
            )
        topk_indices = self.topk_indices_buffer[:num_tokens]
        if topk_indices.dim() == 2:
            topk_indices = topk_indices.unsqueeze(1)
        return topk_indices

    def _forward_attention(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        layer_metadata: AscendDSALayerMetadata,
        cache_is_prepared: bool = False,
    ) -> torch.Tensor:
        # DSA PCP sets cache_is_prepared after global cache updates and forces
        # single-stream attention because there is no local KV update to overlap.
        if cache_is_prepared and self.multistream_dsv4_dsa_overlap:
            raise RuntimeError("Prepared DSA caches require single-stream attention.")

        (
            compress_kv_cache,
            swa_kv_cache,
            state_cache,
            _,
            _,
            _,
        ) = DeviceOperator.unpack_dsa_forward_kv_cache(kv_cache, self.compress_ratio)

        common_attn_metadata = layer_metadata.attention
        if common_attn_metadata is None:
            common_attn_metadata = layer_metadata.swa
        swa_metadata = layer_metadata.swa

        common_metadata = _require_req_metadata(common_attn_metadata)
        swa_req_metadata = _require_req_metadata(swa_metadata)
        has_prefill = common_attn_metadata.num_prefills > 0
        num_tokens = hidden_states.shape[0]
        cos = common_metadata.cos[layer_name][:num_tokens]
        sin = common_metadata.sin[layer_name][:num_tokens]
        actual_seq_lengths_query = common_metadata.query_start_loc
        actual_seq_lengths_key = common_metadata.seq_lens
        ori_win_left = self.window_size - 1 if swa_req_metadata.ori_win_left is None else swa_req_metadata.ori_win_left
        ori_win_right = 0 if swa_req_metadata.ori_win_right is None else swa_req_metadata.ori_win_right

        if self.multistream_dsv4_dsa_overlap:
            q, qr, qr_pertoken_scale = self._mla_prolog_multistream(
                hidden_states,
                cos,
                sin,
                swa_kv_cache,
                swa_req_metadata.slot_mapping,
                is_prefill=has_prefill,
            )
        else:
            q, qr, qr_pertoken_scale = self._mla_prolog_single_stream(
                hidden_states,
                cos,
                sin,
                swa_kv_cache,
                swa_req_metadata.slot_mapping,
                write_swa_cache=not cache_is_prepared,
            )

        compress_topk_idxs = None
        compressor_metadata = None
        if self.has_compressed:
            compressor_metadata = layer_metadata.compressor
            assert compressor_metadata is not None
            compress_topk_idxs = self._maybe_update_compressed_caches_and_select_topk(
                layer_name=layer_name,
                hidden_states=hidden_states,
                qr=qr,
                kv_cache=kv_cache,
                layer_metadata=layer_metadata,
                qr_pertoken_scale=qr_pertoken_scale,
                compress_kv_cache=compress_kv_cache,
                state_cache=state_cache,
                write_cache=not cache_is_prepared,
            )

        notify_kv_cache_written(layer_name)
        wait_for_device_metadata(DeviceMetadataStage.ATTENTION, id(common_metadata.sas_metadata))
        record_attention_compute_start()
        kv_plan = get_dsa_attn_kv_plan(self.vllm_config)
        attn_op = kv_plan.get_dsa_sparse_attn_op()
        attn_kwargs = kv_plan.get_dsa_sparse_attn_base_kwargs()
        if has_prefill:
            kv_plan.add_dsa_sparse_attn_extra_kwargs(attn_kwargs, cu_seqlens_ori_kv=actual_seq_lengths_query)
        if self.has_compressed:
            kv_plan.add_dsa_sparse_attn_extra_kwargs(attn_kwargs, cu_seqlens_cmp_kv=common_metadata.cu_cmp_seqlen_list)

        # graph-decode freshness: the persist set travels INSIDE the req
        # metadata object (same mechanism as sas_metadata) — captured
        # kernels bind these persistent storages, each eager build() rewrites
        # them. Per-group: ori kwargs come from the SWA group's set, cmp
        # kwargs from the compressed group's set. SLICE to this step's row
        # count (the buffers are max_num_seqs wide; the kernels validate
        # seqused/table length against the batch).
        _pb_swa = getattr(swa_req_metadata, "v41_persist", None)
        _pb_cmp = getattr(common_metadata, "v41_persist", None)
        _obt_kw = swa_req_metadata.block_table
        _sq_kw = actual_seq_lengths_key
        if _pb_swa is not None:
            _n_rows = int(actual_seq_lengths_key.shape[0])
            _obt_kw = _pb_swa["ori_bt"][:_n_rows]
            _sq_kw = _pb_swa["seqused"][:_n_rows]
        attn_kwargs.update(
            ori_kv=swa_kv_cache,
            ori_block_table=_obt_kw,
            cu_seqlens_q=actual_seq_lengths_query,
            seqused_kv=_sq_kw,
            sinks=self.attn_sink,
            metadata=common_metadata.sas_metadata,
            softmax_scale=self.softmax_scale,
            cmp_ratio=_dsa_swa_only_cmp_ratio(self.compress_ratio, self.vllm_config, self.has_compressed),
            ori_mask_mode=4,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout_q="TND",
            layout_kv=_dsa_layout_kv(self.vllm_config),
        )

        _v41_swa_only = (
            __import__("os").environ.get("V41_ATTN_SWA_ONLY", "0") == "1"
            and getattr(self, "compress_ratio", 0) in (1, 2)
        )
        if not self.has_compressed or _v41_swa_only:
            if swa_req_metadata.dspark_swa_indices is not None:
                attn_kwargs["ori_sparse_indices"] = swa_req_metadata.dspark_swa_indices
        else:
            assert compressor_metadata is not None
            _cbt_kw = common_metadata.block_table
            if _pb_cmp is not None:
                _cbt_kw = _pb_cmp["cmp_bt"][: int(common_metadata.block_table.shape[0])]
            attn_kwargs.update(
                cmp_kv=compress_kv_cache,
                cmp_block_table=_cbt_kw,
                cmp_mask_mode=3,
            )
            if _dsa_uses_indexer(self.compress_ratio, self.has_compressed):
                assert compress_topk_idxs is not None
                attn_kwargs["cmp_sparse_indices"] = compress_topk_idxs

        if __import__("os").environ.get("V41_GRAPH_STEPDBG", "0") == "1":
            # register per-layer graph-read samples into the persist set the
            # builder drains (eager build); the previous builder-local
            # attribute was never visible to the builder (different objects)
            _persist_dbg = _pb_cmp if _pb_cmp is not None else _pb_swa
            if getattr(self, "_v41_smladbg", None) is None:
                self._v41_smladbg = {}
            sdbg = self._v41_smladbg.get(layer_name)
            if sdbg is None:
                sdbg = {
                    "step": torch.zeros(1, dtype=torch.int64, device=q.device),
                    "seqused": torch.zeros(4, dtype=torch.int64, device=q.device),
                    "ori_bt": torch.zeros(4, dtype=torch.int64, device=q.device),
                    "cmp_bt": torch.zeros(4, dtype=torch.int64, device=q.device),
                    "cmp_idx": torch.full((8,), -3, dtype=torch.int32, device=q.device),
                    "sas": torch.zeros(8, dtype=torch.int32, device=q.device),
                }
                self._v41_smladbg[layer_name] = sdbg
                if _persist_dbg is not None:
                    _persist_dbg["smladbg"][layer_name] = sdbg
            sdbg["step"] += 1
            # sample from the FULL persist buffer (the sliced [:B] kwargs
            # would fail the >=4 guard for B=1 and never update the sample)
            _sq = _pb_swa["seqused"] if _pb_swa is not None else attn_kwargs.get("seqused_kv")
            if _sq is not None and _sq.numel() >= 4:
                sdbg["seqused"][:4] = _sq[:4].reshape(-1).long()
            _obt = attn_kwargs.get("ori_block_table")
            if _obt is not None and _obt.dim() == 2 and _obt.shape[1] >= 4:
                sdbg["ori_bt"][:4] = _obt[0, :4].long()
            _cbt = attn_kwargs.get("cmp_block_table")
            if _cbt is not None and _cbt.dim() == 2 and _cbt.shape[1] >= 4:
                sdbg["cmp_bt"][:4] = _cbt[0, :4].long()
            _cidx = attn_kwargs.get("cmp_sparse_indices")
            if _cidx is not None and _cidx.reshape(-1).numel() >= 8:
                sdbg["cmp_idx"][:8] = _cidx.reshape(-1)[:8].long()
            _sas = attn_kwargs.get("metadata")
            if _sas is not None and _sas.numel() >= 8:
                sdbg["sas"][:8] = _sas[:8]
        if "cmp_sparse_indices" in attn_kwargs:
            # R21 B-side snapshot: what THIS attention call receives.
            # Primary identity from the compressor req metadata; ratio-2
            # layers lack that chain — fall back to the SWA metadata.
            try:
                try:
                    _rm_b = None
                    _cm_md = getattr(layer_metadata, "compressor", None)
                    if _cm_md is not None and getattr(_cm_md, "cache", None) is not None:
                        _rm_b = getattr(_cm_md.cache, "req_metadata", None)
                    if _rm_b is not None:
                        _bB = int(_rm_b.num_actual_reqs) if getattr(_rm_b, "num_actual_reqs", None) is not None else int(_rm_b.seq_lens.shape[0])
                        _sl_b = _rm_b.seq_lens[:_bB].detach().cpu().tolist()
                        _sp_b = (_rm_b.start_pos[:_bB].detach().cpu().tolist()
                                 if getattr(_rm_b, "start_pos", None) is not None
                                 else (_rm_b.seq_lens[:_bB]
                                       - (_rm_b.query_start_loc[1:_bB + 1]
                                          - _rm_b.query_start_loc[:_bB])).detach().cpu().tolist())
                        _fb_b = _rm_b.block_table[:_bB, 0].detach().cpu().tolist()
                        _qsl_b = _rm_b.query_start_loc[: _bB + 1].detach().cpu().tolist()
                    else:
                        raise AttributeError("no compressor req metadata")
                except Exception:
                    _sq_b = actual_seq_lengths_key
                    _qs_b = actual_seq_lengths_query
                    _bB = int(_sq_b.shape[0])
                    _sl_b = _sq_b[:_bB].detach().cpu().tolist()
                    _qsl_b = _qs_b[: _bB + 1].detach().cpu().tolist()
                    _sp_b = (_sq_b[:_bB]
                             - (_qs_b[1:_bB + 1] - _qs_b[:_bB])).detach().cpu().tolist()
                    _fb_b = swa_req_metadata.block_table[:_bB, 0].detach().cpu().tolist()
                try:
                    _pool_b = getattr(self, "candidates_buffer", None)
                    if _pool_b is None:
                        _rt_b = getattr(self, "shared_index_runtime", None)
                        if _rt_b is not None:
                            _pool_b = getattr(_rt_b, "candidates", None)
                except Exception:  # noqa: BLE001 — pool watch must not kill B
                    _pool_b = None
                _v41_ab_capture(
                    "B", layer_name, attn_kwargs["cmp_sparse_indices"],
                    _sl_b, _sp_b, _fb_b, _qsl_b, pool=_pool_b,
                )
            except Exception:  # noqa: BLE001
                pass
        _out = attn_op(q, **attn_kwargs)[0]
        if num_tokens <= 64 and layer_name.endswith(
                ".layers.0.self_attn.attn"):
            _v41_r57_dump("attn_prerope", _out)
        _io_dir = _v41_layer_io_dir()
        if _io_dir is not None:
            _cap = False
            try:
                _cap = torch.npu.is_current_stream_capturing()
            except Exception:  # noqa: BLE001
                _cap = False
            _io = getattr(self, "_v41_layer_io", None)
            if _io is None:
                _io = self._v41_layer_io = {}
            _reg = None
            if _cap and (_pb_cmp is not None or _pb_swa is not None):
                _reg = (_pb_cmp if _pb_cmp is not None else _pb_swa)
            _key, _ent = v41_io_capture(
                _io, layer_name, _out, hidden_states, q, swa_kv_cache,
                swa_req_metadata.slot_mapping, attn_kwargs,
                kvprobe=getattr(self, "_v41_kvprobe", None))
            if _reg is not None and _key + "io" not in _reg["smladbg"]:
                # register EVERY capture: the entry may predate the capture
                # (created by an eager/warmup call) — registering only at
                # creation left replayed steps undrained (cycle56 lesson)
                _reg["smladbg"][_key + "io"] = _ent
            if not _cap:
                _v41_io_write_file(_io_dir, _key, _ent)
        return _out
