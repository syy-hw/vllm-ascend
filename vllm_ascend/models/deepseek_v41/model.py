# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend implementation of DeepSeek V4.1 (DeepseekV41ForCausalLM).

Skeleton copied from ``vllm_ascend/models/deepseek_v4/model.py``; the V4.1
differences implemented here (spec impl-spec.md A2):

- Per-layer ``compress_ratio`` comes from ``compress_ratios`` (0 = pure SWA,
  1/2 = compressed). The compressor exists only on ``kv_source_layer_ids``
  (2/8/14/20); the indexer only on ``index_source_layer_ids``
  (2/8/14/20/24/28/32/36), with ``wk``/``k_norm``/index-K-cache on the four
  kv_source layers ("owns_k").
- Compressed-KV sharing: every compressed non-source layer gets
  ``kv_sharing_target_layer_name`` pointing at the nearest source layer at or
  below it (L3-7->L2, L9-13->L8, L15-19->L14, L21-39->L20), so the runner
  aliases its compressed cache onto the source's (SWA rings stay per-layer).
- topk sharing: one model-level ``topk_indices_buffer``; index sources
  compute and write, other compressed layers read.
- MoE: baseline noaux_tc router plus ``gate.bias_vl`` (vision routing bias).
- Engram on layers 1/14: applied to the residual stream before attention.
- Vision tower + aligner + three delimiter vectors; the multimodal image path
  (processor registration, span embedding assembly, bias_vl routing) lives in
  processing.py plus embed_multimodal/image_span_embedding here.
- No model-level hc_head parameters (absent from the V4.1 checkpoint): the
  final collapse uses the last decoder layer's ffn pre-mix, matching the
  official Transformer.forward.
"""

import typing
from collections.abc import Callable, Iterable
from itertools import islice
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import vllm.envs as envs
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MultiModalEmbeddings,
    SupportsMultiModal,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    _merge_multimodal_embeddings,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

from vllm_ascend.attention.dsa_attn_kv_plan import get_dsv4_attn_kv_dtype
from vllm_ascend.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)
from vllm_ascend.models.deepseek_v4.compressor import Compressor
from vllm_ascend.models.deepseek_v4.model import (
    AscendDeepseekV4ForCausalLM,
    AscendDeepseekV4SWACache,
    DeepseekV4DecoderLayer,
    DeepseekV4MoE,
    get_spec_layer_idx_from_weight_name,
)
from vllm_ascend.models.deepseek_v41.engram import Engram, EngramEmbeddingTable, EngramLayout, NgramHashState
from vllm_ascend.models.deepseek_v41.indexer import AscendDeepseekV41Indexer, SharedIndexRuntime
from vllm_ascend.models.deepseek_v41.processing import (
    IMAGE_PLACEHOLDER,
    DeepseekV41DummyInputsBuilder,
    DeepseekV41MultiModalProcessor,
    DeepseekV41ProcessingInfo,
)
from vllm_ascend.models.deepseek_v41.vision import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_START,
    Aligner,
    ViT,
    image_token_types,
    num_image_tokens,
)
from vllm_ascend.ops.dsa import AscendDeepseekSparseAttention, DSAModules
from vllm_ascend.attention.dsa_v1 import (
    _V41_PERSIST_REGISTRY,
    _v41_io_drain_dir,
    _v41_io_write_file,
    _v41_layer_io_dir,
)
from vllm_ascend.ops.rope_dsv4 import ComplexExpRotaryEmbedding
from vllm_ascend.utils import enable_dsa_cp, extract_dsv4_layer_index, get_dsv4_compress_ratio
from vllm_ascend.worker.v2.pp_utils import (
    PPTransportDataType,
    add_pp_transport_tensors,
    get_pp_transport_tensors,
)
from vllm_ascend.worker.v2.pp_utils import (
    make_empty_intermediate_tensors as make_pp_empty_intermediate_tensors,
)


def nearest_kv_source_layer(layer_idx: int, kv_source_layer_ids: list[int]) -> int | None:
    """Nearest compressed-KV source layer at or below ``layer_idx`` (None if none)."""
    sources = [src for src in kv_source_layer_ids if src <= layer_idx]
    return sources[-1] if sources else None


# ---------------------------------------------------------------------------
# V4.1 numerics layer-dump hook (env V41_DUMP_LAYERS="0,1,2" | "all").
#
# On the first real (non-profile) forward, save input_ids / positions, the
# per-layer output hidden states and the final (pre-logits) hidden states to
# /ws/evidence/numerics/ as .pt files, then disable itself. Profiling /
# warmup forwards use max_num_batched_tokens (>= 4096) tokens, real probe
# prompts are short, so the token-count window below separates them.
# Debug-only: never active unless the env var is set.
# ---------------------------------------------------------------------------
_V41_DUMP = {"layers": None, "done": False, "dir": "/ws/evidence/numerics"}


def _v41_dump_enabled():
    import os

    if _V41_DUMP["layers"] is None:
        env = os.environ.get("V41_DUMP_LAYERS")
        if not env:
            return None
        if env.strip() == "all":
            _V41_DUMP["layers"] = "all"
        else:
            _V41_DUMP["layers"] = {int(x) for x in env.split(",") if x.strip() != ""}
        try:
            os.makedirs(_V41_DUMP["dir"], exist_ok=True)
        except OSError:
            pass
    return _V41_DUMP["layers"] if not _V41_DUMP["done"] else None


def _v41_dump_sublayer(layer_idx: int, tag: str, tensor) -> None:
    """Sub-layer dump (V41_DUMP_SUBLAYERS=1): same selection rules as layers.

    Only fires while the model-level hook has selected THIS forward (the
    ``active`` flag): engine warmup steps run 1-2 token forwards that must
    not consume or pollute the dump."""
    import os

    if (
        os.environ.get("V41_DUMP_SUBLAYERS", "0") != "1"
        or _V41_DUMP["done"]
        or not _V41_DUMP.get("active")
    ):
        return
    layers = _V41_DUMP["layers"]
    if layers is not None and layers != "all" and layer_idx not in layers:
        return
    _v41_dump_tensor(f"L{layer_idx}_{tag}", tensor)


def _v41_dump_tensor(tag: str, tensor) -> None:
    import os

    try:
        rank = os.environ.get("VLLM_DP_RANK") or os.environ.get("LOCAL_RANK") or "0"
        path = os.path.join(_V41_DUMP["dir"], f"{tag}_r{rank}_{os.getpid()}.pt")
        t = tensor.detach()
        # plain D2H memcpy first, cast on CPU (fused cast dispatches to AICPU
        # RunCp, which rejects these dtypes on this stack)
        torch.save({"tensor": t.cpu(), "shape": tuple(t.shape), "dtype": str(t.dtype)}, path)
    except Exception as exc:  # noqa: BLE001 - debug hook must never kill serving
        from vllm.logger import init_logger

        init_logger(__name__).warning("V41_DUMP_LAYERS: failed to save %s: %s", tag, exc)


class DeepseekV41MoE(DeepseekV4MoE):
    """Baseline V4 MoE plus the vision routing bias ``gate.bias_vl``.

    Router semantics stay noaux_tc: the bias (per-token ``bias`` or
    ``bias_vl`` inside image spans) only picks experts, the routing weights
    come from the unbiased scores.
    """

    def __init__(
        self,
        config: DeepseekV41Config,
        parallel_config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_draft_layer: bool = False,
        vision_enabled: bool = False,
    ):
        super().__init__(
            config=config,
            parallel_config=parallel_config,
            quant_config=quant_config,
            prefix=prefix,
            is_draft_layer=is_draft_layer,
        )
        if vision_enabled:
            self.gate.bias_vl = nn.Parameter(torch.empty(config.n_routed_experts, dtype=torch.float32))
        else:
            self.gate.bias_vl = None
        self._v41_topk = config.num_experts_per_tok
        if vision_enabled:
            self._install_vl_router_seam()

    def _install_vl_router_seam(self) -> None:
        """noaux_tc_for_vl routing: expert selection for image-span tokens uses
        ``gate.bias_vl`` instead of ``gate.e_score_correction_bias`` (official
        Gate.forward per-token bias swap); the routing weights stay the
        unbiased scores either way.

        The fused NPU gating op takes one bias per call and the fused MoE
        comm stack expects exactly one moe_forward per layer, so the swap is
        installed as a ``_compute_routing`` seam on the router instance: with
        no active image mask it defers to the original fused path unchanged;
        with one, the whole batch routes through the official torch math
        (per-token bias; identical to the upstream
        ``_topk_softplus_sqrt_torch`` reference incl. renorm and scaling).
        """
        router = getattr(self.experts, "router", None)
        if router is None or getattr(router, "e_score_correction_bias", None) is None:
            return
        orig_compute = router._compute_routing
        moe = self

        def _compute_routing_with_vl(hidden_states, router_logits, indices_type, *, input_ids=None):
            mask = getattr(moe, "_active_image_mask", None)
            if mask is None:
                return orig_compute(hidden_states, router_logits, indices_type, input_ids=input_ids)
            scores = torch.sqrt(F.softplus(router_logits.float()))
            bias = torch.where(
                mask.reshape(-1, 1).to(scores.device),
                moe.gate.bias_vl.float(),
                router.e_score_correction_bias.float(),
            )
            indices = torch.topk(scores + bias, k=router.top_k, dim=-1)[1]
            weights = scores.gather(1, indices)
            if getattr(router, "renormalize", True):
                weights = weights / (weights.sum(dim=-1, keepdim=True).clamp(min=1e-20))
            weights = weights * getattr(router, "routed_scaling_factor", 1.0)
            ids = indices.to(indices_type if indices_type is not None else torch.int32)
            return weights, ids

        router._compute_routing = _compute_routing_with_vl

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image_mask is not None and self.gate.bias_vl is not None:
            return self._forward_with_vl_bias(hidden_states, input_ids, image_mask)
        if __import__("os").environ.get("V41_DUMP_SUBLAYERS", "0") == "1" and _V41_DUMP.get("active"):
            # numerics dump: official router semantics evaluated on the exact
            # engine input (sqrtsoftplus + noaux_tc bias top-k + renorm)
            scores = F.softplus(F.linear(hidden_states.float(), self.gate.weight.float())).sqrt()
            bias = self.gate.e_score_correction_bias
            topk = self._v41_topk
            indices = (scores + bias.float()).topk(topk, dim=-1)[1]
            weights = scores.gather(1, indices)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
            _v41_dump_tensor(f"L{self.layer_idx}_routerscores", scores)
            _v41_dump_tensor(f"L{self.layer_idx}_routeridx", indices)
            _v41_dump_tensor(f"L{self.layer_idx}_routerw", weights)
            # R61: also record the routing the FUSED path actually returns —
            # the reimplementation above is not evidence of what ran.
            _r = getattr(self.experts, "router", None)
            if _r is not None and not getattr(_r, "_v41_r61_wrapped", False):
                _orig_cr = _r._compute_routing
                _layer_idx = self.layer_idx

                def _cr_rec(hidden_states, router_logits, indices_type, *, input_ids=None):
                    w, ids = _orig_cr(hidden_states, router_logits, indices_type,
                                      input_ids=input_ids)
                    try:
                        import os as _o

                        import torch as _t
                        _d = _V41_DUMP["dir"]
                        _rk = _o.environ.get("VLLM_DP_RANK", "0")
                        _pid = _o.getpid()
                        _t.save({"tensor": ids.detach().cpu().long(),
                                 "shape": tuple(ids.shape),
                                 "dtype": "int64"},
                                _o.path.join(_d, f"L{_layer_idx}_routeridx_fused_r{_rk}_{_pid}.pt"))
                        _t.save({"tensor": w.detach().cpu().float(),
                                 "shape": tuple(w.shape), "dtype": "float32"},
                                _o.path.join(_d, f"L{_layer_idx}_routerw_fused_r{_rk}_{_pid}.pt"))
                    except Exception:  # noqa: BLE001
                        pass
                    return w, ids

                _r._compute_routing = _cr_rec
                _r._v41_r61_wrapped = True
        return super().forward(hidden_states, input_ids)

    def _forward_with_vl_bias(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
        image_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Single fused MoE call with the image-span mask stashed for the
        router seam installed by ``_install_vl_router_seam`` (image rows
        select experts with ``gate.bias_vl``; routing weights unchanged).
        Keeps exactly one moe_forward per layer so the fused MoE comm stack
        and the per-step layer counter stay in sync."""
        self._active_image_mask = image_mask.reshape(-1)
        try:
            return super().forward(hidden_states, input_ids)
        finally:
            self._active_image_mask = None


class DeepseekV41Attention(nn.Module):
    """V4.1 latent attention: baseline modules, V4.1 compressor/indexer gating.

    Module/weight layout (wq_a/q_norm/wq_b/wkv/kv_norm/wo_a/wo_b/attn_sink and
    the per-layer SWA ring) is identical to DeepseekV4Attention.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV41Config,
        max_position_embeddings: int = 0,
        cache_config=None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        shared_index_runtime=None,
        reduce_results: bool = True,
        need_gather_q_kv: bool = False,
    ) -> None:
        super().__init__()
        layer_idx = int(prefix.split(sep=".")[-2])
        self.layer_idx = layer_idx
        config_layer_idx = extract_dsv4_layer_index(config, prefix)
        tp_size = get_tensor_model_parallel_world_size()
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_local_heads = config.num_attention_heads // tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = config.head_dim - config.qk_rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.window_size = config.sliding_window
        self.eps = config.rms_norm_eps
        self.norm_eps = config.rms_norm_eps
        self.scale = self.head_dim**-0.5
        self.enable_dsa_cp = enable_dsa_cp()

        # V4.1 layer plan
        self.compress_ratio = get_dsv4_compress_ratio(config, config_layer_idx)
        self.has_compressed = self.compress_ratio > 0
        self.is_kv_source = (
            config_layer_idx < config.num_hidden_layers and config_layer_idx in config.kv_source_layer_ids
        )
        self.is_index_source = (
            config_layer_idx < config.num_hidden_layers and config_layer_idx in config.index_source_layer_ids
        )
        if self.has_compressed and not self.is_kv_source:
            # Shared compressed layer: it reads the nearest source layer's
            # compressed cache and the model-level top-k buffer, so its
            # effective compressed grid is the OWNER's ratio (official configs
            # keep group ratios aligned; the per-layer value would otherwise
            # disagree with the cache/top-k it consumes and with the aclnn
            # CSA kernel's cmp_ratio).
            owner = nearest_kv_source_layer(config_layer_idx, list(config.kv_source_layer_ids))
            if owner is None:
                raise ValueError(
                    f"layer {config_layer_idx} is compressed but has no kv source at or below it"
                )
            owner_ratio = int(get_dsv4_compress_ratio(config, owner))
            self.compress_ratio = owner_ratio

        attn_sink_heads = self.n_heads if self.enable_dsa_cp else self.n_local_heads
        self.attn_sink = nn.Parameter(torch.empty(attn_sink_heads, dtype=torch.float32))
        self.wq_a = ReplicatedLinear(
            self.dim,
            self.q_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_a",
            return_bias=False,
        )
        self.q_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_norm_without_weight = RMSNorm(self.head_dim, eps=config.rms_norm_eps, has_weight=False)
        wq_b_cls = ReplicatedLinear if self.enable_dsa_cp else ColumnParallelLinear
        self.wq_b = wq_b_cls(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
            return_bias=False,
        )
        self.wkv = ReplicatedLinear(
            self.dim,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wkv",
            return_bias=False,
        )
        self.kv_norm = RMSNorm(self.head_dim, self.norm_eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * config.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wo_a",
            return_bias=False,
        )
        self.wo_b = RowParallelLinear(
            self.n_groups * config.o_lora_rank,
            self.dim,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.wo_b",
            return_bias=False,
        )

        # Compressed layers rotate queries at the compressed rope theta (one
        # latent stands for `ratio` tokens, so its positions are further
        # apart); pure SWA layers use the base theta.
        rope_parameters = dict(config.rope_parameters)
        if self.has_compressed:
            rope_parameters["rope_theta"] = config.compress_rope_theta
            rope_groups = ["default", f"c{self.compress_ratio}"]
        else:
            rope_parameters["rope_theta"] = config.rope_theta
            rope_groups = ["default"]
        self.rotary_emb = ComplexExpRotaryEmbedding(
            vllm_config=vllm_config,
            layername=f"{prefix}.attn",
            head_size=self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position_embeddings=max_position_embeddings,
            is_neox_style=False,
            scaling_factor=rope_parameters["factor"],
            base=rope_parameters["rope_theta"],
            beta_fast=rope_parameters["beta_fast"],
            beta_slow=rope_parameters["beta_slow"],
            rope_groups=rope_groups,
        )

        # Compressor only on kv_source layers; V4.1 ships no ape weights.
        self.compressor: Compressor | None = None
        self.indexer: AscendDeepseekV41Indexer | None = None
        if self.is_kv_source:
            self.compressor = Compressor(
                vllm_config,
                config,
                self.compress_ratio,
                head_dim=self.head_dim,
                quant_config=quant_config,
                cache_config=cache_config,
                prefix=f"{prefix}.compressor",
                no_ape=True,
            )
        if self.is_index_source:
            self.indexer = AscendDeepseekV41Indexer(
                vllm_config,
                config,
                self.compress_ratio,
                owns_k=self.is_kv_source,
                quant_config=quant_config,
                cache_config=cache_config,
                prefix=f"{prefix}.indexer",
            )

        k_dtype = get_dsv4_attn_kv_dtype(vllm_config)
        swa_cache_layer = AscendDeepseekV4SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=k_dtype,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
        )

        dsa_modules = DSAModules(
            wq_a=self.wq_a,
            q_norm=self.q_norm,
            q_norm_without_weight=self.q_norm_without_weight,
            wq_b=self.wq_b,
            wkv=self.wkv,
            kv_norm=self.kv_norm,
            wo_a=self.wo_a,
            wo_b=self.wo_b,
            attn_sink=self.attn_sink,
            indexer=self.indexer,
            compressor=self.compressor,
            swa_cache_layer=swa_cache_layer,
        )

        self.dsa_attn = AscendDeepseekSparseAttention(
            dim=self.dim,
            n_heads=self.n_heads,
            scale=self.scale,
            n_local_heads=self.n_local_heads,
            q_lora_rank=self.q_lora_rank,
            o_lora_rank=self.o_lora_rank,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            nope_head_dim=self.nope_head_dim,
            eps=self.eps,
            n_groups=self.n_groups,
            n_local_groups=self.n_local_groups,
            window_size=self.window_size,
            compress_ratio=self.compress_ratio,
            dsa_modules=dsa_modules,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}",
            need_gather_q_kv=need_gather_q_kv,
            has_compressed=self.has_compressed,
            topk_indices_buffer=topk_indices_buffer,
            shared_index_runtime=shared_index_runtime,
            # Official V4.1 Attention has no per-head q RMSNorm after wq_b;
            # the q_norm_without_weight RMS the V4 baseline applies is a V4-only
            # step and measurably deviates from the official math (numerics
            # capture: q rel ~0.5-0.8 with it, ~0.01 without).
            apply_q_head_rms=False,
        )

        # Compressed-KV sharing (vllm generic mechanism): non-source layers
        # reuse the nearest source layer's compressed cache. The attribute
        # lives on the spec-owning DSAAttention module ("...self_attn.attn"),
        # the per-layer SWA ring stays private.
        # Always record the owning kv-source layer name on the inner
        # DSAAttention: the dsa_v1 bridge and the ops/dsa.py _build_kv_cache
        # fallback use it to read the SOURCE's compressed cache when this
        # layer's own binding is absent. VLLM_V41_DISABLE_KV_SHARING=1 only
        # turns off the generic runner-level cache GROUPING (whose
        # init_attn_backend path does not understand this model's per-layer
        # SWA/compress split and KeyErrors on 'model.layers.21...'); with
        # grouping off the shared layer still allocates its own (unused)
        # cache and resolves the source cache at forward time.
        if self.has_compressed and not self.is_kv_source:
            target = nearest_kv_source_layer(config_layer_idx, list(config.kv_source_layer_ids))
            if target is None:
                raise ValueError(
                    f"layer {config_layer_idx} is compressed but has no kv source at or below it"
                )
            self_name = f"{prefix}.attn"
            target_name = self_name.replace(f".layers.{layer_idx}.", f".layers.{target}.")
            # v41_kv_source_layer_name: consumed by the ops/dsa.py
            # _build_kv_cache fallback to read the SOURCE's compressed cache
            # when this layer's own binding is absent. Deliberately NOT the
            # generic kv_sharing_target_layer_name attribute: the generic
            # runner-level cache grouping (init_attn_backend) does not
            # understand this model's per-layer SWA/compress split
            # (KeyError 'model.layers.21.self_attn.attn'), so that stays
            # gated behind VLLM_V41_DISABLE_KV_SHARING!=1.
            self.dsa_attn.dsa_attn.v41_kv_source_layer_name = target_name
            import os as _os

            if _os.environ.get("VLLM_V41_DISABLE_KV_SHARING", "0") != "1":
                self.dsa_attn.dsa_attn.kv_sharing_target_layer_name = target_name

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.dsa_attn(positions, hidden_states, llama_4_scaling)


class DeepseekV41DecoderLayer(DeepseekV4DecoderLayer):
    """V4.1 block: baseline hc handling, V4.1 attention/MoE, optional engram."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config: DeepseekV41Config | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        shared_index_runtime=None,
        is_draft_layer: bool = False,
        engram_layout: EngramLayout | None = None,
    ) -> None:
        # NOTE: deliberately not calling DeepseekV4DecoderLayer.__init__ (it
        # builds the V4 attention); we reconstruct the same attribute set so
        # isinstance-based MoE plumbing keeps working.
        nn.Module.__init__(self)

        if config is None:
            config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = config.rope_parameters["original_max_position_embeddings"]
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx
        self.norm_eps = config.rms_norm_eps
        self.use_sequence_parallel_moe = parallel_config.use_sequence_parallel_moe
        self.enable_dsa_cp = enable_dsa_cp()

        self.self_attn = DeepseekV41Attention(
            vllm_config=vllm_config,
            config=config,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
            shared_index_runtime=shared_index_runtime,
            reduce_results=not self.use_sequence_parallel_moe,
            need_gather_q_kv=self.use_sequence_parallel_moe and self.enable_dsa_cp,
        )

        self.mlp = DeepseekV41MoE(
            config=config,
            parallel_config=parallel_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            is_draft_layer=is_draft_layer,
            vision_enabled=getattr(config, "vision_enabled", False),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=self.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=self.norm_eps)
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

        # Engram (layers 1/14): n-gram lookup added to the residual stream
        # before attention. Its fp8 table is NOT a parameter (CPU mmap), the
        # wkv/q_weight/k_weight parameters load normally.
        self.engram: Engram | None = None
        if engram_layout is not None and layer_idx in engram_layout.layer_ids:
            self.engram = Engram(
                config,
                layer_idx,
                engram_layout,
                quant_config=quant_config,
                prefix=f"{prefix}.engram",
            )

    def hc_mixes(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor,
                 hc_base: torch.Tensor):
        """Official Block.hc_mixes: one projection of the flattened stream,
        normalized over hc*hidden per token, split through the official
        sinkhorn into (pre, post, comb). x is [tokens, hc_mult, hidden]."""
        xf = x.flatten(1).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(xf, hc_fn) * rsqrt
        return self._hc_split_sinkhorn(mixes, hc_scale, hc_base)

    @staticmethod
    def make_identity_pre_mix(hidden_states: torch.Tensor, hc_mult: int) -> torch.Tensor:
        """Official make_identity_pre_mix: one-hot on the first hc copy."""
        pre_mix = hidden_states.new_zeros(hidden_states.shape[0], hc_mult, dtype=torch.float32)
        pre_mix[:, 0] = 1.0
        return pre_mix

    def hc_pre_collapse(self, x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        """Official Block.hc_pre: collapse the hc copies weighted by ``pre_mix``."""
        y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=1)
        return y.to(x.dtype)

    def _hc_split_sinkhorn(self, mixes: torch.Tensor, hc_scale: torch.Tensor,
                           hc_base: torch.Tensor):
        """Official hc_split_sinkhorn (kernel.py) in pure torch.

        mixes: [n, mix_hc] fp32 -> pre [n, hc], post [n, hc], comb [n, hc, hc]
        """
        hc, eps = self.hc_mult, self.hc_eps
        pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
        post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1]
                                 + hc_base[hc:2 * hc])
        comb = (mixes[..., 2 * hc:].unflatten(-1, (hc, hc)) * hc_scale[2]
                + hc_base[2 * hc:].view(hc, hc))
        comb = comb.softmax(dim=-1) + eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        return pre, post, comb

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor,
                comb: torch.Tensor) -> torch.Tensor:
        """Torch-native npu_hc_post: expand out and mix the residual in.

        x: [tokens, hidden], residual: [tokens, hc, hidden], post: [tokens, hc],
        comb: [tokens, hc, hc] -> [tokens, hc, hidden].
        """
        y = (post.unsqueeze(-1) * x.unsqueeze(-2)
             + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=1))
        return y.type_as(x)

    def _v41_eng_layer_probe_enabled(self):
        import os

        return os.environ.get("V41_ENG_LAYER_PROBE", "0") == "1"

    def _v41_eng_layer_probe_record(self, read_rows, post_hidden,
                                   write_file=True):
        """Record the engram rows THIS execution read + the post-engram
        hidden into per-layer persistent buffers (device copies — safe
        inside the compiled region, refreshed per replay). The NEXT
        v41_engram_graph_hook drains them (post-replay ordering); eager
        steps drain via the io-capture file write below."""
        probe = getattr(self, "_v41_eng_layer_probe_buf", None)
        t = min(read_rows.shape[0], 4)
        flat = post_hidden[:t].reshape(t, -1)
        if probe is None:
            # hidden at the engram point is [T, hc_mult, dim]: allocate the
            # out buffer at the ACTUAL flattened width on first record
            probe = self._v41_eng_layer_probe_buf = {
                "read": torch.zeros(
                    4, *read_rows.shape[1:], dtype=read_rows.dtype,
                    device=read_rows.device),
                "out": torch.zeros(
                    4, flat.shape[-1], dtype=post_hidden.dtype,
                    device=post_hidden.device),
                "n": 0,
            }
        probe["read"][:t] = read_rows[:t]
        probe["out"][:t] = flat
        probe["n"] += 1
        if not write_file:
            return  # graph/compiled path: the hook drains post-replay
        import os as _os

        d = _os.environ.get("V41_ENG_LAYER_DUMP_DIR", "")
        if d:
            try:
                _os.makedirs(d, exist_ok=True)
                torch.save(
                    {"layer": self.layer_idx, "n": probe["n"],
                     "read": probe["read"].detach().cpu(),
                     "out": probe["out"].detach().cpu(),
                     "pid": _os.getpid()},
                    _os.path.join(
                        d, f"englyr_L{self.layer_idx}_"
                           f"p{_os.getpid()}_{probe['n']:06d}.pt"))
            except Exception:  # noqa: BLE001
                pass

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        engram_hashes: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        pre_mix: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """``pre_mix`` is the PREVIOUS layer's ffn pre-mix (None = identity,
        official make_identity_pre_mix): it collapses the attention input.
        Official Block.forward order: the attention input collapses with the
        previous ffn pre-mix; the FFN input collapses with THIS layer's attn
        pre; the returned ffn pre-mix feeds the next layer's attention (and
        the final collapse for the last layer)."""
        if self.engram is not None and engram_hashes is not None:
            layer_hashes = engram_hashes[:, self.engram.layer_hash_index, :]
            # Official semantics: image-span tokens get no engram contribution
            # (engram_mask = ~image_mask).
            engram_token_mask = None if image_mask is None else ~image_mask
            _eng_io = _v41_eng_io_enabled()
            embedded_probe = self.engram.embed.lookup(
                layer_hashes, device=hidden_states.device) \
                if (_eng_io or self._v41_eng_layer_probe_enabled()) else None
            _pre_h = hidden_states
            hidden_states = self.engram(hidden_states, layer_hashes, token_mask=engram_token_mask)
            _v41_eng_bump("full", hidden_states.device)
            if _eng_io and embedded_probe is not None:
                _v41_eng_io_update(self, positions, embedded_probe,
                                   _pre_h, hidden_states)
                _v41_eng_io_write_now(self)
            if embedded_probe is not None:
                self._v41_eng_layer_probe_record(
                    embedded_probe, hidden_states)
        elif self.engram is not None and engram_hashes is None:
            if torch.compiler.is_compiling():
                # decode-graph capture: the official computation runs SPLIT —
                # host hash + CPU-mmap gather in the eager prepare hook
                # (v41_engram_graph_hook), device remainder here, reading the
                # persistent per-layer buffer (allocated at init, stable
                # storage). Skipping this would silently drop the engram
                # contribution from every replayed decode step.
                buf = getattr(self, "_v41_engram_emb_buf", None)
                if buf is not None:
                    T = hidden_states.shape[0]
                    if T > buf.shape[0]:
                        raise RuntimeError(
                            f"engram staging buffer too small for the captured "
                            f"step: {T} tokens > buffer {buf.shape[0]}")
                    _read_rows = buf[:T]
                    _pre_h = hidden_states
                    hidden_states = self.engram.apply_embedded(hidden_states, _read_rows)
                    _v41_eng_bump("apply", hidden_states.device)
                    if _v41_eng_io_enabled():
                        _v41_eng_io_update(self, positions, _read_rows,
                                           _pre_h, hidden_states)
                    if self._v41_eng_layer_probe_enabled():
                        # snapshot WHAT THIS EXECUTION ACTUALLY READ (device
                        # copy inside the compiled region: refreshes on every
                        # replay) and the post-engram hidden — not a
                        # reference to the possibly-overwritten buffer
                        self._v41_eng_layer_probe_record(
                            _read_rows, hidden_states, write_file=False)
            elif __import__("os").environ.get("V41_SKIP_ENGRAM", "0") != "1":
                raise NotImplementedError(
                    "engram is enabled but no hash ids were provided; "
                    "NgramHashState integration is deferred (Pass B)"
                )
            # V41_SKIP_ENGRAM=1: pass the stream through untouched (smoke).

        if pre_mix is None:
            pre_mix = self.make_identity_pre_mix(hidden_states, self.hc_mult)
        residual = hidden_states.clone()
        full_num_tokens = positions.shape[0]
        attn_pre, attn_post, attn_comb = self.hc_mixes(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        hidden_states = self.hc_pre_collapse(hidden_states, pre_mix)
        hidden_states = self.input_layernorm(hidden_states)
        _v41_dump_sublayer(self.layer_idx, "attnin", hidden_states)

        if self.use_sequence_parallel_moe and not self.enable_dsa_cp:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]

        attn_kwargs = {"positions": positions, "hidden_states": hidden_states, "llama_4_scaling": llama_4_scaling}
        hidden_states = self.self_attn(**attn_kwargs)
        _v41_dump_sublayer(self.layer_idx, "attnout", hidden_states)

        if self.use_sequence_parallel_moe and not self.enable_dsa_cp:
            hidden_states = sp_reduce_scatter(hidden_states)

        hidden_states = self.hc_post(hidden_states, residual, attn_post, attn_comb)

        residual = hidden_states.clone()
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        hidden_states = self.hc_pre_collapse(hidden_states, attn_pre)
        hidden_states = self.post_attention_layernorm(hidden_states)
        _v41_dump_sublayer(self.layer_idx, "ffnin", hidden_states)
        hidden_states = self.mlp(hidden_states, input_ids, image_mask=image_mask)
        _v41_dump_sublayer(self.layer_idx, "mlpout", hidden_states)
        hidden_states = self.hc_post(hidden_states, residual, ffn_post, ffn_comb)

        return hidden_states, residual, ffn_pre


# V4.1 engram branch probe (V41_ENG_PROBE=1): DEVICE counters bumped inside
# each DecoderLayer engram branch. Under FULL ACL graph capture the bump is
# a recorded device op, so the counter that grows per REPLAY identifies
# WHICH branch the captured graph actually contains ("full" = the whole
# Engram.forward traced at capture; "apply" = apply_embedded reading the
# hook-staged buffer). v41_engram_graph_hook prints the counters each FULL
# step (eager Python), revealing the deltas of the LAST replay.
_V41_ENG_PROBE: dict = {}


def _v41_eng_bump(kind: str, dev) -> None:
    import os

    if os.environ.get("V41_ENG_PROBE", "0") != "1":
        return
    t = _V41_ENG_PROBE.get(kind)
    if t is None or t.device != dev:
        t = _V41_ENG_PROBE[kind] = torch.zeros(
            1, dtype=torch.int64, device=dev)
    t += 1


_V41_ENG_DUMP: dict = {"n": 0}


def _v41_eng_io_mode() -> str:
    """V41_ENG_IO switch semantics (source-verified, R17): unset/"0" = off,
    "1" = full content capture (engread/engpre/engpost copies), "2"/"lite"/
    "identity" = identity-only lite record (step/ntok/seq/pos int64 tensors
    only — no tensor content). The value is baked into the compiled
    artifacts and the init pre-allocation, so changing it requires a
    service restart."""
    import os

    v = os.environ.get("V41_ENG_IO", "0")
    if v == "1":
        return "full"
    if v in ("2", "lite", "identity"):
        return "lite"
    return "off"


def _v41_eng_io_enabled() -> bool:
    return _v41_eng_io_mode() != "off"


def _v41_eng_io_update(layer, positions, read_rows, pre_hidden,
                       post_hidden) -> None:
    """Traced-safe engram-site capture update (review-15): pure device ops
    only — a DEVICE step counter, T, positions[0] (identity: advances
    44,45,... on decode) and the first min(T,4) rows this execution READ
    plus the pre/post-apply hidden. NO stream-capture query and NO file
    writes here: dynamo cannot trace torch_npu.is_current_stream_capturing
    inside the strictly-compiled backbone (live crash at profile run).
    Graph-side files come from the post-replay drain (v41_io_drain_all,
    entry registered host-side by the hook) plus the hook-top drain that
    also covers compiled-not-captured steps (prefill). The eager path
    writes at forward time via _v41_eng_io_write_now (untraced).
    """
    try:
        ent = getattr(layer, "_v41_eng_io_entry", None)
        if _v41_eng_io_mode() == "lite":
            # lite record (R17 heavy-collection off): identity only — the
            # four int64 tensors below. NO content copies: engread/engpre/
            # engpost stay absent, the payload writer emits None for them.
            if ent is None or "engread" in ent:
                dev = read_rows.device
                ent = layer._v41_eng_io_entry = {
                    "step": torch.zeros(1, dtype=torch.int64, device=dev),
                    "ntok": torch.zeros(1, dtype=torch.int64, device=dev),
                    "seq": torch.zeros(1, dtype=torch.int64, device=dev),
                    "pos": torch.zeros(4, dtype=torch.int64, device=dev),
                }
            ent["step"] += 1
            ent["ntok"].fill_(int(read_rows.shape[0]))
            _p = positions.reshape(-1)
            ent["seq"][0] = _p[0].long()
            _np = min(4, int(_p.numel()))
            ent["pos"][:_np] = _p[:_np].long()
            return
        t = min(int(read_rows.shape[0]), 4)
        flat_w = int(pre_hidden.reshape(pre_hidden.shape[0], -1).shape[-1])
        if ent is None or ent["engpre"].shape[-1] != flat_w:
            dev = read_rows.device
            ent = layer._v41_eng_io_entry = {
                "step": torch.zeros(1, dtype=torch.int64, device=dev),
                "ntok": torch.zeros(1, dtype=torch.int64, device=dev),
                "seq": torch.zeros(1, dtype=torch.int64, device=dev),
                "pos": torch.zeros(4, dtype=torch.int64, device=dev),
                "engread": torch.zeros(
                    4, *read_rows.shape[1:], dtype=read_rows.dtype,
                    device=dev),
                "engpre": torch.zeros(
                    4, flat_w, dtype=pre_hidden.dtype, device=dev),
                "engpost": torch.zeros(
                    4, flat_w, dtype=post_hidden.dtype, device=dev),
            }
        ent["step"] += 1
        ent["ntok"].fill_(int(read_rows.shape[0]))
        _p = positions.reshape(-1)
        ent["seq"][0] = _p[0].long()
        _np = min(4, int(_p.numel()))
        ent["pos"][:_np] = _p[:_np].long()
        ent["engread"][:t] = read_rows[:t]
        ent["engpre"][:t] = pre_hidden[:t].reshape(t, -1)
        ent["engpost"][:t] = post_hidden[:t].reshape(t, -1)
    except Exception:  # noqa: BLE001 - debug hook must never kill serving
        pass


def _v41_eng_io_write_now(layer) -> None:
    """Untraced (eager server) path: write the entry to a file at forward
    time. Never called from inside the compiled backbone. Uses the drain
    dir fallback (V41_LAYER_IO_DIR, else V41_ENG_IO_DIR) so the lite
    record covers eager prefill steps too; dummy-classified content
    (1-token batch at position 0) is never written (same content gate as
    the post-replay drain)."""
    try:
        from vllm_ascend.attention.dsa_v1 import _v41_io_ent_is_dummy
        ent = getattr(layer, "_v41_eng_io_entry", None)
        if ent is not None and not _v41_io_ent_is_dummy(ent):
            _d = _v41_io_drain_dir()
            if _d is not None:
                _v41_io_write_file(_d, f"engramL{layer.layer_idx}", ent)
    except Exception:  # noqa: BLE001
        pass


def _v41_eng_io_hook(model) -> None:
    """Host-side hook step (called from v41_engram_graph_hook, untraced):
    (1) idempotently register every engram layer's entry into the persist
    registry so the aclgraph runner's post-replay drain (v41_io_drain_all)
    writes one file per real replay including the LAST step; (2) in full
    mode, drain the CURRENT entry content — which belongs to the PREVIOUS
    execution, so compiled-not-captured steps (graph-side prefill) also get
    files. In lite mode the hook-top drain is skipped: coverage comes from
    the eager write_now (prefill) and the gated post-replay drain (every
    real replay); dummy-only content is never written."""
    try:
        _lite = _v41_eng_io_mode() == "lite"
        for layer in model.layers:
            if getattr(layer, "engram", None) is None:
                continue
            ent = getattr(layer, "_v41_eng_io_entry", None)
            if ent is None:
                continue
            for _reg in _V41_PERSIST_REGISTRY:
                if "smladbg" in _reg:
                    _reg["smladbg"][f"engramL{layer.layer_idx}_io"] = ent
                    break
            if _lite:
                continue
            _d = _v41_layer_io_dir()
            if _d is not None:
                _v41_io_write_file(_d, f"engramL{layer.layer_idx}", ent)
    except Exception:  # noqa: BLE001
        pass


def _v41_engram_dump_write(site, hashes, ids, qsl, seq_lens, keys, positions, T,
                           staged=None):
    """Full-hash dump with capture identity (review-13).

    One file per record: {dir}/eng_{site}_r{rank}_p{pid}_s{n:06d}.pt with a
    per-process counter — workers never overwrite each other. The payload
    carries the full identity (token ids, positions, per-request keys,
    seq_lens, T, phase); comparators bind records by payload, never by
    filename sort order. Gated by V41_DUMP_ENGRAM=1, dir via
    V41_ENG_DUMP_DIR (default /ws/evidence/engram-dump).
    """
    import os

    try:
        d = os.environ.get("V41_ENG_DUMP_DIR", "") or "/ws/evidence/engram-dump"
        os.makedirs(d, exist_ok=True)
        n = _V41_ENG_DUMP["n"]
        _V41_ENG_DUMP["n"] = n + 1
        rank = os.environ.get("VLLM_DP_RANK", os.environ.get("LOCAL_RANK", "0"))
        pos = positions.detach().cpu() if torch.is_tensor(positions) else None
        fname = f"eng_{site}_r{rank}_p{os.getpid()}_{n:06d}.pt"
        fpath = os.path.join(d, fname)
        if os.path.exists(fpath):
            # duplicate write must never silently overwrite (review-14):
            # disambiguate with a monotonic suffix and keep BOTH records
            k = 1
            while os.path.exists(fpath[:-3] + f"_{k}.pt"):
                k += 1
            fpath = fpath[:-3] + f"_{k}.pt"
        torch.save(
            {"site": site, "rank": rank, "pid": os.getpid(), "n": n,
             "hashes": hashes.detach().cpu(),
             "ids": ids.detach().cpu() if torch.is_tensor(ids)
             else torch.tensor(list(ids)),
             "qsl": torch.tensor(list(qsl) if qsl is not None else []),
             "seq_lens": torch.tensor(
                 list(seq_lens) if seq_lens is not None else []),
             "keys": list(keys) if keys is not None else [],
             "positions": pos, "T": int(T),
             "phase": "prefill" if int(T) > 1 else "decode",
             "staged": staged},
            fpath)
    except Exception as exc:  # noqa: BLE001 - debug hook must not kill serving
        from vllm.logger import init_logger

        init_logger(__name__).warning("V41_DUMP_ENGRAM: save failed: %s", exc)


@support_torch_compile
class DeepseekV41Model(nn.Module, EagleModelMixin):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type
        self.use_sequence_parallel_moe = vllm_config.parallel_config.use_sequence_parallel_moe

        self.vocab_size = config.vocab_size
        topk_tokens = config.index_topk
        topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            topk_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        # Exposed at model level so spec_decode/llm_base_proposer can share
        # this buffer with the MTP draft via attribute replacement.
        self.topk_indices_buffer = topk_indices_buffer
        # Model-level hand-off for the V4.1 reference indexer path (official
        # SharedAttentionRuntime): owns_k layers publish an index-K provider
        # reading their reference store, and the candidate source layer (20)
        # publishes its candidate-block mask; non-owns_k index sources
        # (24/28/32/36) consume both.
        self.shared_index_runtime = SharedIndexRuntime()

        self.engram_layout = EngramLayout.from_args(config)
        self.engram_hash: NgramHashState | None = None
        if self.engram_layout is not None:
            # Tokenizer binding (compressed token map) happens at serve time.
            self.engram_hash = NgramHashState(config, self.engram_layout, tokenizer=None)

        if get_pp_group().is_first_rank:
            # embed.weight ships bf16 with no fp8 scale in the V4.1
            # checkpoint -> unquantized path (unlike V4-Flash).
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=None,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV41DecoderLayer(
                vllm_config,
                prefix,
                topk_indices_buffer=topk_indices_buffer,
                shared_index_runtime=self.shared_index_runtime,
                engram_layout=self.engram_layout,
            ),
            prefix=f"{prefix}.layers",
        )
        # Graph-decode engram staging: one persistent buffer per engram layer
        # (stable storage allocated HERE, before any capture — the captured
        # graph binds it once). The eager prepare hook (v41_engram_graph_hook,
        # called by the v2 model runner on FULL-graph decode steps) computes
        # the official per-step hashes on host, gathers the CPU-mmap table
        # rows, and rewrites the buffer contents; the captured graph runs the
        # device remainder (Engram.apply_embedded). Without this the compile
        # path silently dropped the engram contribution entirely.
        # Row bound = max_num_batched_tokens: the profile/compile dummy run
        # executes a full-size batch (4096 tokens) under compiling=True and
        # its outputs feed memory accounting, so the buffer must cover that
        # shape too (~2 layers x 4096 x 24 x 256 x 2B = 100MB/rank).
        self._v41_engram_state = {"emb": {}}
        if self.engram_layout is not None:
            _max_t = max(
                int(vllm_config.scheduler_config.max_num_seqs),
                int(vllm_config.scheduler_config.max_num_batched_tokens),
            )
            for _layer in self.layers:
                _eng = getattr(_layer, "engram", None)
                if _eng is None or isinstance(_layer, PPMissingLayer):
                    continue
                self._v41_engram_state["emb"][_layer.layer_idx] = torch.zeros(
                    _max_t, self.engram_layout.n_hash_cols, _eng.embed.head_dim,
                    dtype=torch.bfloat16,
                    device=torch.device("npu", torch.npu.current_device()),
                )
                _layer._v41_engram_emb_buf = self._v41_engram_state["emb"][_layer.layer_idx]
                # Pre-allocate the engram io-entry AND dict-probe at INIT
                # (graph_el5 lesson): anything lazily allocated inside a
                # traced artifact gets the None-branch BAKED into that
                # artifact — each later artifact execution then allocates a
                # NEW entry and re-binds the module attribute, while the
                # captured decode graph keeps writing the ORPHANED warmup
                # tensors. Drains follow the attribute (frozen content,
                # colliding filenames). Pre-allocation makes every traced
                # artifact bake the no-alloc branch over stable tensors —
                # the exact property that keeps emb_buf replay-fresh.
                import os as _os4e

                _dev = torch.device("npu", torch.npu.current_device())
                _flat = int(getattr(_layer, "hc_mult", 4)) * self.config.hidden_size
                if _v41_eng_io_mode() == "full":
                    _layer._v41_eng_io_entry = {
                        "step": torch.zeros(1, dtype=torch.int64, device=_dev),
                        "ntok": torch.zeros(1, dtype=torch.int64, device=_dev),
                        "seq": torch.zeros(1, dtype=torch.int64, device=_dev),
                        "pos": torch.zeros(4, dtype=torch.int64, device=_dev),
                        "engread": torch.zeros(
                            4, self.engram_layout.n_hash_cols,
                            _eng.embed.head_dim, dtype=torch.bfloat16,
                            device=_dev),
                        "engpre": torch.zeros(
                            4, _flat, dtype=torch.bfloat16, device=_dev),
                        "engpost": torch.zeros(
                            4, _flat, dtype=torch.bfloat16, device=_dev),
                    }
                elif _v41_eng_io_mode() == "lite":
                    _layer._v41_eng_io_entry = {
                        "step": torch.zeros(1, dtype=torch.int64, device=_dev),
                        "ntok": torch.zeros(1, dtype=torch.int64, device=_dev),
                        "seq": torch.zeros(1, dtype=torch.int64, device=_dev),
                        "pos": torch.zeros(4, dtype=torch.int64, device=_dev),
                    }
                if _os4e.environ.get("V41_ENG_LAYER_PROBE", "0") == "1":
                    _layer._v41_eng_layer_probe_buf = {
                        "read": torch.zeros(
                            4, self.engram_layout.n_hash_cols,
                            _eng.embed.head_dim, dtype=torch.bfloat16,
                            device=_dev),
                        "out": torch.zeros(
                            4, _flat, dtype=torch.bfloat16, device=_dev),
                        "n": 0,
                    }
        # V4.1 msProbe dump hook (env-gated, default OFF): wraps the module
        # forwards of the requested scope (V41_MSPROBE_SCOPE, comma list of
        # module-name prefixes, default the first two decoder layers) with
        # mindstudio-probe's PrecisionDebugger. NOTE for interpretation:
        # under FULL ACL graphs the Python hooks fire only on EAGER steps
        # (prefill/warmup/capture) — replayed decode steps produce NO module
        # dumps; graph replay data comes from the V41_LAYER_IO post-replay
        # drain instead.
        import os as _os46
        _ms_dir = _os46.environ.get("V41_MSPROBE_DIR", "")
        if _ms_dir:
            try:
                import json as _json

                from msprobe.pytorch import PrecisionDebugger
                _scope = _os46.environ.get(
                    "V41_MSPROBE_SCOPE", "layers.0,layers.1")
                _steps = int(_os46.environ.get("V41_MSPROBE_STEPS", "400"))
                _rank = _os46.environ.get("VLLM_DP_RANK", "0")
                # FLAT schema + rank=None + level L0, verified against
                # mindstudio-probe 26.2.0 on this worker (nested "dump"
                # section / rank:[0] / level "module" are all rejected)
                _dump_root = _os46.path.join(_ms_dir, f"rank{_rank}")
                _os46.makedirs(_dump_root, exist_ok=True)
                _cfg_path = _os46.path.join(_ms_dir, f"cfg_p{_os46.getpid()}.json")
                with open(_cfg_path, "w") as _fh:
                    _fh.write(_json.dumps({
                        "task": _os46.environ.get("V41_MSPROBE_TASK",
                                                  "tensor"),
                        "dump_path": _dump_root,
                        "rank": None,
                        "step": [0, _steps],
                        "level": "L0",
                        "scope": [x for x in _scope.split(",") if x] or None,
                        "list": None,
                    }))
                self._v41_msdbg = PrecisionDebugger(config_path=_cfg_path)
                self._v41_msdbg.start(self)
                print(f"[V41_MSPROBE] started rank={_rank} scope={_scope} "
                      f"steps<={_steps}", flush=True)
            except Exception as _e:  # noqa: BLE001 - instrumentation only
                import traceback as _tb
                print(f"[V41_MSPROBE] init failed: {_e!r}\n"
                      + _tb.format_exc(), flush=True)
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        def make_empty_intermediate_tensors(batch_size: int, dtype: torch.dtype, device: torch.device):
            return IntermediateTensors(
                {
                    "hidden_states": torch.zeros(
                        (batch_size, self.hc_mult, config.hidden_size),
                        dtype=dtype,
                        device=device,
                    ),
                }
            )

        self.make_empty_intermediate_tensors = make_pp_empty_intermediate_tensors(
            self,
            make_empty_intermediate_tensors,
        )

        self.norm_eps = config.rms_norm_eps
        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        hc_dim = hc_mult * config.hidden_size

        # No hc_head parameters in V4.1 (absent from the checkpoint): the
        # final collapse reuses the last decoder layer's ffn pre-mix; see
        # forward() and DeepseekV41DecoderLayer.compute_ffn_pre_mix.

        # DSpark/MTP off: no draft model is built, mtp.* weights are skipped.

        spec_config = vllm_config.speculative_config
        needs_mtp_hidden_states = spec_config is not None and (
            spec_config.use_eagle() or spec_config.uses_draft_model()
        )
        self._mtp_hidden_buffer = (
            torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                hc_dim,
                dtype=vllm_config.model_config.dtype,
                device=self.device,
            )
            if get_pp_group().is_last_rank and needs_mtp_hidden_states
            else None
        )

        # Vision tower + aligner + delimiters. The multimodal processor is
        # registered on the ForCausalLM class (see processing.py): official
        # image_processor grid planning (min_pixels 295936, max 1024 tokens)
        # expanding each image_token_id placeholder into its span; the span
        # embeddings come back through embed_multimodal/image_span_embedding.
        self.vision: ViT | None = None
        self.aligner: Aligner | None = None
        if getattr(config, "vision_enabled", False):
            vision_config = SimpleNamespace(**config.vision_config)
            self.vision = ViT(vision_config)
            self.aligner = Aligner(vision_config, text_dim=config.hidden_size)
            self.image_start = nn.Parameter(torch.empty(config.hidden_size))
            self.image_end = nn.Parameter(torch.empty(config.hidden_size))
            self.image_newline = nn.Parameter(torch.empty(config.hidden_size))
        self.image_token_id = config.image_token_id

    @property
    def shared_index_k_provider(self):
        """Callable reading the most recent owns_k layer's reference index-K store."""
        return self.shared_index_runtime.index_k_provider

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def bind_engram_tokenizer(self, checkpoint_dir: str) -> None:
        """Bind the engram hash state's compressed token map (tokenizer).

        Official NgramHashState is built with the served tokenizer; here the
        tokenizer is only available on the API-server side, so the map is
        rebuilt from the checkpoint's tokenizer.json at load time. The
        compressed-vocab size check inside bind_tokenizer is the hard guard
        against a mismatching tokenizer rehashing the whole table.
        """
        import os

        if self.engram_hash is None:
            return
        tok_path = os.path.join(checkpoint_dir, "tokenizer.json")
        if not os.path.exists(tok_path):
            raise FileNotFoundError(f"engram needs a tokenizer.json under {checkpoint_dir}")
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=False)
        except Exception:
            from tokenizers import Tokenizer

            backend = Tokenizer.from_file(tok_path)

            class _BackendShim:
                def __init__(self, backend):
                    self.backend_tokenizer = backend
                    self._n = backend.get_vocab_size(with_added_tokens=True)

                def __len__(self):
                    return self._n

            tokenizer = _BackendShim(backend)
        self.engram_hash.bind_tokenizer(tokenizer)


    def _v41_engram_hashes(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_image: torch.Tensor | None = None,
    ):
        """Hash ids for this step's tokens: [T, n_engram_layers, n_hash_cols].

        Maintains the per-request compressed-id history (official
        NgramHashState): prefill builds it fully, decode appends one token per
        request. Request identity is the first KV block id (stable for a
        request's lifetime, same key the indexer reference store uses).
        ``is_image`` (True inside image spans) keeps image tokens out of the
        n-gram history (official engram_mask). Returns None when engram is
        skipped or the tokenizer is unbound.
        """
        import os

        state = self.engram_hash
        if state is None or state.token_map is None:
            return None
        if os.environ.get("V41_SKIP_ENGRAM", "0") == "1":
            return None

        T = input_ids.shape[0]
        ids_cpu = input_ids.detach().cpu().long()

        req_metadata = None
        if is_forward_context_available():
            try:
                ctx = get_forward_context()
                am = getattr(ctx, "attn_metadata", None)
            except Exception:
                am = None
            if isinstance(am, dict):
                for value in am.values():
                    rm = getattr(value, "req_metadata", None)
                    if rm is not None and getattr(rm, "query_start_loc", None) is not None:
                        req_metadata = rm
                        break
            elif am is not None and getattr(am, "query_start_loc", None) is not None:
                req_metadata = am

        if torch.compiler.is_compiling():
            # graph capture/replay: the hash path is host code (CPU
            # histories, .cpu() reads); decode replay runs without engram
            # (official gate = no-op without fresh hashes)
            return None
        if req_metadata is None:
            # Profile / warmup forwards carry no attention metadata: hash the
            # flattened batch as one synthetic request WITHOUT touching the
            # persistent histories (their outputs are discarded anyway).
            from vllm_ascend.models.deepseek_v41.engram import compute_ngram_hashes

            row = ids_cpu.reshape(1, -1)
            scratch = state._history_row("__v41_profile__")
            saved_len = state._lengths.get("__v41_profile__", 0)
            state.extend("__v41_profile__", row.reshape(-1), None)
            n = row.shape[1]
            hashes = compute_ngram_hashes(
                row,
                torch.arange(0, n, dtype=torch.int64, device="cpu").unsqueeze(0),
                state._history["__v41_profile__"].unsqueeze(0),
                state.multipliers,
                state.primes,
                state.offsets,
                state.pad_id,
                state.DEAD,
            )
            # do not let the scratch row grow unboundedly
            state._history.pop("__v41_profile__", None)
            state._lengths.pop("__v41_profile__", None)
            return hashes.reshape(T, -1, state.layout.n_hash_cols).to(input_ids.device)

        qsl = req_metadata.query_start_loc.detach().cpu().tolist()
        seq_lens = req_metadata.seq_lens.detach().cpu().tolist()
        n_actual = getattr(req_metadata, "num_actual_reqs", None)
        B = len(seq_lens)
        if n_actual is not None:
            B = min(int(n_actual), B)
        block_table = getattr(req_metadata, "block_table", None)
        # history key: engine request slot via the persist channel when
        # available — the SAME identity v41_engram_graph_hook keys by, so a
        # request alternating eager steps and graph steps keeps one history.
        # Fallback (no persist set built): the first KV block id.
        _persist = getattr(req_metadata, "v41_persist", None)
        if _persist is not None:
            first_blocks = _persist["req_slot"][:B].detach().cpu().tolist()
        elif block_table is not None:
            first_blocks = block_table[:B, 0].detach().cpu().tolist()
        else:
            first_blocks = list(range(B))
        is_image_cpu = None
        if is_image is not None:
            is_image_cpu = is_image.reshape(-1).detach().cpu()

        parts = []
        for i in range(B):
            t0, t1 = int(qsl[i]), int(qsl[i + 1])
            if t1 <= t0:
                continue
            start_pos = int(seq_lens[i]) - (t1 - t0)
            req_key = int(first_blocks[i])
            img_slice = None if is_image_cpu is None else is_image_cpu[t0:t1]
            # Align the keyed history before appending: block ids get reused
            # (eviction) or shared (prefix caching) across requests, and a
            # stale row must not be read as this request's tokens.
            state.seek(req_key, start_pos)
            state.extend(req_key, ids_cpu[t0:t1], img_slice)
            hashes = state.hashes([req_key], [start_pos], [t1 - t0])
            parts.append(hashes[0])  # [L, n_layers, n_hash_cols]
        if not parts:
            return None
        out = torch.cat(parts, dim=0)  # [T, n_layers, n_hash_cols] token order
        import os as _os

        if _os.environ.get("V41_DUMP_ENGRAM", "0") == "1":
            _v41_engram_dump_write(
                "eager", out, ids_cpu, qsl[: B + 1],
                seq_lens[:B], first_blocks[:B], positions, out.shape[0])
        return out.to(input_ids.device)

    def _v41_eng_layer_probe_drain(self):
        """Post-replay drain of the per-layer engram probes (called at the
        TOP of every hook: the buffers then hold the LAST executed
        forward's reads — the previous replay, real or dummy). One file per
        drain per layer, named by a HOST-side counter: every executed
        replay is preserved (dummies included — matched out by content at
        analysis time)."""
        import os

        d = os.environ.get("V41_ENG_LAYER_DUMP_DIR", "")
        if not d:
            return
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            return
        n = _V41_ENG_DUMP.get("drain_n", 0)
        _V41_ENG_DUMP["drain_n"] = n + 1
        for layer in self.layers:
            probe = getattr(layer, "_v41_eng_layer_probe_buf", None)
            if probe is None:
                continue
            try:
                torch.save(
                    {"layer": layer.layer_idx, "drain_n": n,
                     "read": probe["read"].detach().cpu(),
                     "out": probe["out"].detach().cpu(),
                     "pid": os.getpid()},
                    os.path.join(
                        d, f"englyrD_L{layer.layer_idx}_p{os.getpid()}_"
                           f"{n:06d}.pt"))
            except Exception:  # noqa: BLE001
                pass

    def v41_engram_graph_hook(self, input_batch, max_tokens: int, req_slots_cpu) -> None:
        self._v41_eng_layer_probe_drain()
        if _v41_eng_io_enabled():
            _v41_eng_io_hook(self)
        """Eager per-step hook for FULL-graph decode steps (v2 model runner
        prepare_inputs calls it BEFORE the graph replays).

        Runs the official engram computation that cannot live inside the
        captured graph: the per-step n-gram hashes from the per-request
        compressed-id history (NgramHashState semantics) and the CPU-mmap
        table row gather. The gathered rows land in the persistent per-layer
        buffers allocated at init; the captured graph then runs the device
        remainder (Engram.apply_embedded). Histories are keyed by the ENGINE
        REQUEST SLOT — the same identity _v41_engram_hashes uses when the
        persist channel carries it — so eager steps and graph steps share one
        history row per request even when a request alternates paths.
        """
        import os

        state = self.engram_hash
        if state is None or state.token_map is None:
            return
        if os.environ.get("V41_SKIP_ENGRAM", "0") == "1":
            return
        T = int(input_batch.num_tokens)
        if T <= 0 or T > int(max_tokens):
            return
        ids = input_batch.input_ids.detach().cpu().long()
        qsl_np = input_batch.query_start_loc_np
        npr = int(input_batch.num_reqs)
        nct = input_batch.num_computed_tokens_np
        parts = []
        for i in range(npr):
            t0, t1 = int(qsl_np[i]), int(qsl_np[i + 1])
            if t1 <= t0:
                continue
            start_pos = int(nct[i])
            key = int(req_slots_cpu[i])
            state.seek(key, start_pos)
            state.extend(key, ids[t0:t1], None)  # decode: no image tokens
            parts.append(state.hashes([key], [start_pos], [t1 - t0])[0])
        if not parts:
            return
        out = torch.cat(parts, dim=0)  # [T_real, n_layers, n_hash_cols]
        dev = input_batch.input_ids.device
        import os

        _staged = None
        if os.environ.get("V41_DUMP_ENGRAM", "0") == "1":
            # graph-side lookup output + the exact rows staged into each
            # layer buffer this step (review-15: eager lookup results
            # cannot substitute for graph-side write evidence)
            _staged = {}
        for layer in self.layers:
            eng = getattr(layer, "engram", None)
            if eng is None:
                continue
            buf = self._v41_engram_state["emb"].get(layer.layer_idx)
            if buf is None or buf.device != dev:
                continue
            emb = eng.embed.lookup(out[:, eng.layer_hash_index, :], device=dev)
            buf[: emb.shape[0]].copy_(emb)
            if _staged is not None:
                _staged[layer.layer_idx] = emb.detach().cpu()

        if os.environ.get("V41_DUMP_ENGRAM", "0") == "1":
            _v41_engram_dump_write(
                "hook", out, ids,
                [int(x) for x in qsl_np[: npr + 1]],
                [int(x) for x in input_batch.seq_lens_np[:npr]]
                if getattr(input_batch, "seq_lens_np", None) is not None
                else None,
                [int(x) for x in req_slots_cpu[:npr]],
                input_batch.positions, out.shape[0], staged=_staged)
        if os.environ.get("V41_ENG_PROBE", "0") == "1":
            # counters are DEVICE tensors bumped inside the captured graph:
            # their growth since the previous hook call = which engram branch
            # the LAST replay actually executed
            _f = _V41_ENG_PROBE.get("full")
            _a = _V41_ENG_PROBE.get("apply")
            _b0 = next(iter(self._v41_engram_state["emb"].values()), None)
            print(f"[ENGPROBE] hook: full={int(_f.item()) if _f is not None else 0} "
                  f"apply={int(_a.item()) if _a is not None else 0} "
                  f"staged0={float(_b0[0].float().abs().sum()) if _b0 is not None else -1.0} "
                  f"ids0={int(input_batch.input_ids.reshape(-1)[0])} "
                  f"hash0={out[0, 0, :4].tolist()} T={int(out.shape[0])}",
                  flush=True)

    def encode_image(self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        assert self.vision is not None and self.aligner is not None
        return self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

    def image_span_embedding(self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        """Assemble the full image-span embedding: [span_len, hidden].

        Official ``merge_image_embeddings`` layout: the IMAGE rows take the
        aligner output in reading order, IMAGE_START / IMAGE_END /
        IMAGE_NEW_LINE rows take the learned delimiter vectors. Built in the
        embedding step (``embed_multimodal``) instead of inside forward; the
        runner scatters the result over the span positions.
        """
        assert self.vision is not None and self.aligner is not None
        r = self.aligner.downsample_ratio
        n_llm_h = -(-n_vit_h // r)
        n_llm_w = -(-n_vit_w // r)
        types = image_token_types(n_llm_h, n_llm_w).to(patches.device)
        embeds = self.encode_image(patches, n_vit_h, n_vit_w)
        dtype = self.image_start.dtype
        span = torch.empty(
            types.numel(), embeds.shape[-1], dtype=dtype, device=patches.device
        )
        span[types == IMAGE_START] = self.image_start.to(dtype)
        span[types == IMAGE_END] = self.image_end.to(dtype)
        span[types == IMAGE_NEW_LINE] = self.image_newline.to(dtype)
        span[types == IMAGE] = embeds.to(dtype)
        return span

    def merge_image_embeddings(self, hidden_states: torch.Tensor, image_inputs) -> torch.Tensor:
        """Overwrite each image's token span in the embedded inputs.

        Official semantics: IMAGE slots take aligner rows in reading order,
        the span delimiters take the learned embeddings. Only valid for a
        prefill chunk (image spans must be prefetched in one piece).
        """
        assert self.vision is not None
        from vllm_ascend.models.deepseek_v41.vision import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START

        h = hidden_states
        for image_input in image_inputs or ():
            types = image_input.types.to(h.device)
            span = h[image_input.start : image_input.start + types.numel()]
            span[types == IMAGE_START] = self.image_start.to(h.dtype)
            span[types == IMAGE_END] = self.image_end.to(h.dtype)
            span[types == IMAGE_NEW_LINE] = self.image_newline.to(h.dtype)
            embeds = self.encode_image(image_input.patches.to(h.device), image_input.n_vit_h, image_input.n_vit_w)
            span[types == IMAGE] = embeds.to(h.dtype)
        return h

    def hc_collapse(self, hidden_states: torch.Tensor, ffn_pre_mix: torch.Tensor) -> torch.Tensor:
        """Collapse the hc copies with the last layer's ffn pre-mix.

        x: [T, hc_mult, dim], ffn_pre_mix: [T, hc_mult] -> [T, dim].
        """
        y = torch.sum(ffn_pre_mix.unsqueeze(-1) * hidden_states.float(), dim=1)
        return y.to(hidden_states.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        engram_hashes: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        pp_group = get_pp_group()
        # V4.1 numerics dump hook: capture the first short (real-request)
        # forward. See _V41_DUMP above.
        _dump_layers = _v41_dump_enabled()
        _dumping = (
            _dump_layers is not None
            and pp_group.is_first_rank
            and input_ids is not None
            and 3 < input_ids.shape[0] < 256  # engine warmups run 1-2 token steps
        )
        _V41_DUMP["active"] = bool(_dumping)
        if _dumping:
            _v41_dump_tensor("input_ids", input_ids)
            if positions is not None:
                _v41_dump_tensor("positions", positions)
        if pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            # Multimodal: image spans were already overwritten at the
            # embedding level (ForCausalLM.embed_multimodal + the runner's
            # scatter); forward only routes the span mask (bias_vl / engram).
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = None
        aux_hidden_states = get_pp_transport_tensors(
            intermediate_tensors,
            PPTransportDataType.AUX_HIDDEN_STATES,
        )

        # V4.1 vision: image-span mask from input_ids (official
        # image_mask = token_types >= 0; every span position carries
        # image_token_id). None when this step has no image tokens so
        # text-only steps keep the unmodified fast path. Computed before the
        # sequence-parallel MoE shard of input_ids (SP + images unsupported).
        if image_mask is None and input_ids is not None and self.vision is not None:
            image_hit = input_ids == self.image_token_id
            # graph-safe: no host bool(); route the mask unconditionally —
            # text-only steps get an all-False mask, which the consumers
            # treat as "no image spans" (bias_vl rows stay zeroed).
            if True:
                if self.use_sequence_parallel_moe:
                    raise NotImplementedError(
                        "multimodal input is not supported with sequence-parallel MoE"
                    )
                image_mask = image_hit

        if self.use_sequence_parallel_moe:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(forward_context.is_padding, hidden_states)
            hidden_states = sp_shard(hidden_states)
            input_ids = sp_shard(input_ids)  # TODO: support PP with dsacp.

        llama_4_scaling = None

        # V4.1 engram wiring: hash the step's tokens against the per-request
        # compressed-id history (official NgramHashState semantics; image
        # tokens are kept out of the n-gram history via is_image). No-op
        # (None) when engram is absent, skipped (V41_SKIP_ENGRAM=1) or the
        # tokenizer is unbound; sequence-parallel MoE shards input_ids, so it
        # is not supported on this path yet.
        if engram_hashes is None and input_ids is not None and not self.use_sequence_parallel_moe:
            engram_hashes = self._v41_engram_hashes(input_ids, positions, is_image=image_mask)

        if pp_group.is_first_rank:
            hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)  # (b, s, h) -> (b, s, c, h)
        pre_mix = None  # first layer collapses with the official identity mix
        final_ffn_pre_mix = None
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual, ffn_pre_mix = layer(
                positions,
                hidden_states,
                residual,
                llama_4_scaling,
                input_ids=input_ids,
                engram_hashes=engram_hashes,
                image_mask=image_mask,
                pre_mix=pre_mix,
            )
            if ffn_pre_mix is not None:
                pre_mix = ffn_pre_mix
                final_ffn_pre_mix = ffn_pre_mix
            if _dumping and (_dump_layers == "all" or layer.layer_idx in _dump_layers):
                _v41_dump_tensor(f"layer{layer.layer_idx}", hidden_states)
            if layer.layer_idx + 1 in self.aux_hidden_state_layers:
                aux_hidden_state = hidden_states.mean(dim=1)
                if self.use_sequence_parallel_moe:
                    aux_hidden_state = sp_all_gather(aux_hidden_state)[: positions.shape[0]]
                aux_hidden_states.append(aux_hidden_state)

        if not pp_group.is_last_rank:
            intermediate_tensors = IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                }
            )
            return add_pp_transport_tensors(
                intermediate_tensors,
                PPTransportDataType.AUX_HIDDEN_STATES,
                aux_hidden_states,
            )

        if self.use_sequence_parallel_moe:
            hidden_states = sp_all_gather(hidden_states)[: positions.shape[0]]

        # Stash pre-collapse residual for the MTP draft (captured copy_).
        if self._mtp_hidden_buffer is not None:
            num_tokens = hidden_states.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

        # Final collapse: official V4.1 uses the last layer's ffn pre-mix
        # (the V4.1 checkpoint has no hc_head weights).
        if final_ffn_pre_mix is not None:
            hidden_states = self.hc_collapse(hidden_states, final_ffn_pre_mix)

        hidden_states = self.norm(hidden_states)
        if _dumping:
            _v41_dump_tensor("final_hidden", hidden_states)
            if final_ffn_pre_mix is not None:
                _v41_dump_tensor("final_ffn_pre_mix", final_ffn_pre_mix)
            _V41_DUMP["done"] = True
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states


def map_v41_checkpoint_name(name: str) -> tuple[str | None, bool]:
    """Map one checkpoint weight name to its model-side parameter name.

    Pure function (no torch) so it can be diffed against the checkpoint
    offline. Returns ``(mapped_name, skip)``; ``skip=True`` means "recognized
    but intentionally not loaded" (mtp.* / engram tables).
    """
    bare = name[len("model."):] if name.startswith("model.") else name

    if bare.startswith("mtp."):
        return None, True  # DSpark off: no draft model built

    # Engram fp8 tables stay CPU-side mmaps; never route through the loader.
    if ".engram.embed.weight" in bare or ".engram.embed.scale" in bare:
        return None, True

    # Vision / aligner / delimiter names short-circuit BEFORE the generic
    # chain: "vision.patch_embed.proj.weight" contains "embed." and
    # "vision.blocks.M.attn.*" contains "attn", both of which the generic
    # chain would rewrite.
    if bare.startswith(("vision.", "aligner.", "image_start", "image_end", "image_newline")):
        return f"model.{bare}", False

    name = bare
    if not name.startswith("model"):
        name = f"model.{name}"

    if ".w1." in name:
        name = name.replace(".w1.", ".gate_proj.")
    if ".w2." in name:
        name = name.replace(".w2.", ".down_proj.")
    if ".w3." in name:
        name = name.replace(".w3.", ".up_proj.")

    if "model.head." in name and "model.lm_head." not in name:
        name = name.replace("model.head.", "lm_head.")
    if "model.lm_head." in name:
        name = name.replace("model.lm_head.", "lm_head.")
    if "embed." in name and "embed_tokens." not in name:
        name = name.replace("embed.", "embed_tokens.")
    if "attn" in name and "self_attn" not in name:
        name = name.replace(".attn.", ".self_attn.")
    if ".ffn." in name:
        name = name.replace(".ffn.", ".mlp.")
    if ".ffn_norm." in name:
        name = name.replace(".ffn_norm.", ".post_attention_layernorm.")
    if ".attn_norm." in name:
        name = name.replace(".attn_norm.", ".input_layernorm.")
    if name.endswith(".scale"):
        name = name.replace(".scale", ".weight_scale")

    if "rotary_emb.inv_freq" in name:
        return "", True  # non-parameter artifact

    # .gate.bias_vl must be matched before the .gate.bias substring rule.
    if ".gate.bias_vl" in name:
        return name, False
    if ".gate.bias" in name:
        name = name.replace(".gate.bias", ".gate.e_score_correction_bias")
    return name, False


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekV41MultiModalProcessor,
    info=DeepseekV41ProcessingInfo,
    dummy_inputs=DeepseekV41DummyInputsBuilder,
)
class AscendDeepseekV41ForCausalLM(
    AscendDeepseekV4ForCausalLM,
    SupportsMultiModal,
):
    """V4.1: V4 body with the V4.1 config/loading described in the module docstring.

    Multimodal (image) input follows the official image_processor/model flow:
    the registered processor expands each ``<｜deepseek_image｜>`` placeholder
    into its image span, ``embed_multimodal`` runs ViT+aligner and assembles
    the span embeddings (delimiters included) and the model runner scatters
    them over the span positions; forward then only routes the span mask
    (``bias_vl`` expert routing, engram exclusion)."""

    model_cls = DeepseekV41Model
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    requires_raw_input_tokens = True
    """forward() needs raw input_ids even in multimodal (inputs_embeds) mode:
    the image-span mask (bias_vl / engram routing) is derived from them."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # NOTE: nn.Module.__init__ directly -- AscendDeepseekV4ForCausalLM's
        # own __init__ would build the V4 model; only its non-init machinery
        # (MoE plumbing, PP/LoRA interfaces) is inherited.
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = self.model_cls(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if get_pp_group().is_last_rank:
            # head.weight ships bf16 with no fp8 scale in the V4.1 checkpoint
            # -> unquantized path (unlike V4-Flash).
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=None,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        # Set MoE hyperparameters
        self.num_moe_layers = self.config.num_hidden_layers
        self.set_moe_parameters()

    # ------------------------------------------------------------------
    # Multimodal (image) input
    # ------------------------------------------------------------------

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return IMAGE_PLACEHOLDER
        raise ValueError("Only image modality is supported")

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Run the vision tower + aligner per image.

        Receives the processor's batched image fields (``image_patches``:
        flat [total_patches, 3, p, p] split via ``image_grids`` [n, 2]) and
        returns one embedding per image span -- the official
        ``merge_image_embeddings`` layout with the learned delimiter rows
        included, in prompt order.
        """
        image_patches = kwargs.get("image_patches")
        image_grids = kwargs.get("image_grids")
        if image_patches is None or image_grids is None:
            raise ValueError(f"Unsupported multimodal kwargs: {sorted(kwargs)}")
        model = self.model
        if model.vision is None or model.aligner is None:
            raise RuntimeError("vision tower not built but image inputs arrived")

        spans: list[torch.Tensor] = []
        offset = 0
        for row in image_grids.tolist():
            n_vit_h, n_vit_w = int(row[0]), int(row[1])
            patches = image_patches[offset : offset + n_vit_h * n_vit_w]
            offset += n_vit_h * n_vit_w
            spans.append(model.image_span_embedding(patches, n_vit_h, n_vit_w))
        return spans

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.embed_input_ids(input_ids)
        if multimodal_embeddings is None or is_multimodal is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal.to(inputs_embeds.device, non_blocking=True),
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        # Image spans are already merged into inputs_embeds by
        # embed_input_ids; the model derives the span mask from input_ids
        # (requires_raw_input_tokens=True keeps them flowing here).
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )
        return hidden_states

    def bind_engram_tables(self, checkpoint_path: str) -> dict[str, str]:
        """Open CPU mmaps for every engram fp8 table found under ``checkpoint_path``.

        The tables never flow through the device loader (load_weights skips
        ``*.engram.embed.weight`` / ``*.embed.scale``); this binds them so
        Engram.forward can gather rows on demand. Returns the bound tensor
        names. Multiple DP ranks pointing at the same file share physical
        pages.
        """
        import glob
        import os

        from safetensors import safe_open

        tables: dict[str, EngramEmbeddingTable] = {}
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if getattr(layer, "engram", None) is not None:
                tables[f"layers.{layer.layer_idx}.engram.embed.weight"] = layer.engram.embed
        bound: dict[str, str] = {}
        if not tables:
            return bound
        files = (
            sorted(glob.glob(os.path.join(checkpoint_path, "*.safetensors")))
            if os.path.isdir(checkpoint_path)
            else [checkpoint_path]
        )
        for file_path in files:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                keys = [key[len("model."):] if key.startswith("model.") else key for key in f.keys()]
                for key in keys:
                    table = tables.get(key)
                    if table is None:
                        continue
                    scale_key = key[: -len(".weight")] + ".scale"
                    table.bind(file_path, key, scale_key)
                    bound[key] = file_path
            if len(bound) == len(tables):
                break
        missing = set(tables) - set(bound)
        if missing:
            raise FileNotFoundError(f"engram tables not found in checkpoint: {sorted(missing)}")
        return bound

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping

        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self.model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        # Attention heads per rank
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank

        skipped: list[str] = []
        for name, loaded_weight in weights:
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model

            name, skip = map_v41_checkpoint_name(name)
            if skip:
                if name:
                    skipped.append(name)
                continue
            if not name:
                continue

            if "sink" in name:
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                if enable_dsa_cp():
                    param.data.copy_(loaded_weight)
                else:
                    # Handle attention sinks (distributed across ranks)
                    narrow_weight = loaded_weight.narrow(0, head_start, heads_per_rank)
                    param.data.copy_(narrow_weight)
                loaded_params.add(name)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if name_mapped not in params_dict:
                    continue
                name = name_mapped
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue

                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)

        if skipped:
            from vllm.logger import init_logger

            init_logger(__name__).info(
                "DeepseekV41: skipped %d non-device tensors (engram fp8 tables / mtp): %s%s",
                len(skipped),
                ", ".join(sorted(skipped)[:8]),
                " ..." if len(skipped) > 8 else "",
            )
        if getattr(self.model, "engram_layout", None) is not None:
            import os

            if os.environ.get("V41_SKIP_ENGRAM", "0") != "1":
                ckpt_dir = getattr(self.config, "_name_or_path", "") or ""
                if not os.path.isdir(ckpt_dir):
                    raise RuntimeError(
                        f"engram enabled but the checkpoint dir is unavailable: {ckpt_dir!r}; "
                        "load_weights cannot bind the fp8 tables"
                    )
                bound = self.bind_engram_tables(ckpt_dir)
                self.model.bind_engram_tokenizer(ckpt_dir)
                from vllm.logger import init_logger

                init_logger(__name__).info(
                    "DeepseekV41: bound engram fp8 tables (%s) and the compressed token map",
                    ", ".join(f"{k} <- {os.path.basename(v)}" for k, v in sorted(bound.items())),
                )
        return loaded_params
