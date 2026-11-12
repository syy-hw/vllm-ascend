# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionMetadata

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.models.layer.attention.layer import DSAAttention


@dataclass
class DSAModules:
    """Modules used in SFA V2."""

    wq_a: torch.nn.Module
    q_norm: torch.nn.Module
    q_norm_without_weight: torch.nn.Module
    wq_b: torch.nn.Module
    wkv: torch.nn.Module
    kv_norm: torch.nn.Module
    wo_a: torch.nn.Module
    wo_b: torch.nn.Module
    attn_sink: torch.nn.Module
    indexer: torch.nn.Module | None
    compressor: torch.nn.Module | None
    swa_cache_layer: torch.nn.Module
    indexer_rotary_emb: torch.nn.Module | None = None


class AscendDeepseekSparseAttention(MultiHeadLatentAttentionWrapper):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        scale: float,
        n_local_heads: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int | None,
        nope_head_dim: int,
        eps: float,
        n_groups: int,
        n_local_groups: int,
        window_size: int,
        compress_ratio: int,
        dsa_modules: DSAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        need_gather_q_kv: bool = False,
        has_compressed: bool | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        shared_index_runtime: Any | None = None,
        # V4.1: official Attention applies NO per-head RMSNorm on q after
        # wq_b; False skips the V4-baseline q head RMS. Default True (V4).
        apply_q_head_rms: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        self.dim = dim
        self.n_heads = n_heads
        self.scale = scale
        self.n_local_heads = n_local_heads
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.nope_head_dim = nope_head_dim
        self.eps = eps
        self.n_groups = n_groups
        self.n_local_groups = n_local_groups
        self.window_size = window_size
        self.compress_ratio = compress_ratio
        # DeepSeek V4.1: ratio 1 must be told apart from pure SWA (spec B3).
        self.has_compressed = (compress_ratio > 1) if has_compressed is None else bool(has_compressed)
        self.need_gather_q_kv = need_gather_q_kv

        self.wq_a = dsa_modules.wq_a
        self.q_norm = dsa_modules.q_norm
        self.q_norm_without_weight = dsa_modules.q_norm_without_weight
        self.wq_b = dsa_modules.wq_b
        self.wkv = dsa_modules.wkv
        self.kv_norm = dsa_modules.kv_norm
        self.wo_a = dsa_modules.wo_a
        self.wo_b = dsa_modules.wo_b
        self.attn_sink = dsa_modules.attn_sink
        self.indexer = dsa_modules.indexer
        self.compressor = dsa_modules.compressor
        self.indexer_rotary_emb = dsa_modules.indexer_rotary_emb
        self.prefix = prefix

        self.swa_cache_layer = dsa_modules.swa_cache_layer

        self.dsa_attn = DSAAttention(
            dim=self.dim,
            n_heads=self.n_heads,
            scale=self.scale,
            n_local_heads=self.n_local_heads,
            q_lora_rank=self.q_lora_rank,
            o_lora_rank=self.o_lora_rank,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            nope_head_dim=self.nope_head_dim,
            n_groups=self.n_groups,
            n_local_groups=self.n_local_groups,
            window_size=self.window_size,
            compress_ratio=self.compress_ratio,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            # extra
            wq_a=self.wq_a,
            wq_b=self.wq_b,
            wkv=self.wkv,
            q_norm=self.q_norm,
            q_norm_without_weight=self.q_norm_without_weight,
            kv_norm=self.kv_norm,
            indexer=self.indexer,
            compressor=self.compressor,
            wo_a=self.wo_a,
            wo_b=self.wo_b,
            attn_sink=self.attn_sink,
            eps=self.eps,
            swa_cache_layer=self.swa_cache_layer,
            has_compressed=self.has_compressed,
            kv_sharing_target_layer_name=getattr(self, "kv_sharing_target_layer_name", None),
            topk_indices_buffer=topk_indices_buffer,
            # V4.1: model-level hand-off (index-K provider + candidates) that
            # owns_k layers publish and non-owns_k index sources consume.
            shared_index_runtime=shared_index_runtime,
            apply_q_head_rms=apply_q_head_rms,
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        # Patch 71: keep a direct handle so _build_kv_cache can resolve the
        # V4.1 kv-source layer's cache without get_current_vllm_config(),
        # which is unset in the dummy-batch/warmup forward contexts.
        self._vllm_static_forward_context = compilation_config.static_forward_context

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor | None = None,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        output_shape = hidden_states.shape

        output = torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)

        # All DSA forward paths (attention + o_proj, including OTP HCCL
        # collectives) run inside the dsa_forward custom op, which is required
        # for ACL graph capture (registered with dispatch_key="PrivateUse1").
        torch.ops.vllm.dsa_forward(hidden_states, self.need_gather_q_kv, output, self.prefix)

        output = output.view(-1, output_shape[-1])
        return output


@eager_break_during_capture
def dsa_forward(
    hidden_states: torch.Tensor,
    need_gather_q_kv: bool,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    attn_metadata = forward_context.attn_metadata

    if attn_metadata is None:
        # Profiling run: forward() handles OTP by running _forward_o_proj on a
        # zero input so HCCL collectives are captured by the ACL graph.
        if need_gather_q_kv:
            self.dsa_attn.impl.forward(
                self.dsa_attn.layer_name,
                hidden_states,
                None,
                None,
                output,
                need_gather_q_kv=True,
            )
        else:
            self.dsa_attn.impl.forward(self.dsa_attn.layer_name, hidden_states, None, None, output)
        return

    kv_cache = _build_kv_cache(self, forward_context)

    if need_gather_q_kv:
        self.dsa_attn.impl.forward(
            self.dsa_attn.layer_name,
            hidden_states,
            kv_cache,
            attn_metadata,
            output,
            need_gather_q_kv=True,
        )
    else:
        self.dsa_attn.impl.forward(self.dsa_attn.layer_name, hidden_states, kv_cache, attn_metadata, output)
    # R56: one-shot dump of L0's attention output BEFORE the output
    # projection, on the same first-real-forward the model-level layer
    # dumps select. Bisects Q/KV->window-attention vs o_proj for the
    # eager-prefill L0 divergence.
    try:
        from vllm_ascend.models.deepseek_v41.model import _V41_DUMP as _D
        if (_D.get("active") and not _D.get("attn0_done")
                and self.dsa_attn.layer_name.endswith(".layers.0.self_attn.attn")):
            _D["attn0_done"] = True
            import os as _os

            import torch as _t
            _rank = _os.environ.get("VLLM_DP_RANK", "0")
            _base = _os.path.join(_D["dir"], f"L0_pid{_os.getpid()}_r{_rank}")
            _t.save({"tensor": output.detach().cpu(),
                     "shape": tuple(output.shape),
                     "dtype": str(output.dtype)},
                    _base + "_attnpost_o_proj.pt")
            _opi = getattr(self.dsa_attn.impl, "o_proj_input", None)
            if _opi is not None:
                _t.save({"tensor": _opi.detach().cpu(),
                         "shape": tuple(_opi.shape),
                         "dtype": str(_opi.dtype)},
                        _base + "_attnpre_o_proj.pt")
    except Exception:  # noqa: BLE001 - probe must never kill serving
        pass
    return


def dsa_forward_fake(
    hidden_states: torch.Tensor,
    need_gather_q_kv: bool,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="dsa_forward",
    op_func=dsa_forward,
    mutates_args=["output"],
    fake_impl=dsa_forward_fake,
    dispatch_key="PrivateUse1",
)


def _build_kv_cache(self, forward_context):
    """Construct the 6-tuple KV cache used by impl.forward()."""
    compress_kv_cache = None
    swa_kv_cache = self.swa_cache_layer.kv_cache
    state_cache = None
    indexer_state_cache = None
    indexer_k_cache = None
    indexer_scale_cache = None
    indexer_full_cache = None

    if self.has_compressed:
        if self.compressor is not None and self.compressor.state_cache is not None:
            state_cache = self.compressor.state_cache.kv_cache
        # A V4.1 shared compressed layer has compressor=None and reads the
        # compressed KV through the shared dsa_attn.kv_cache binding below.
        compress_kv_cache = self.dsa_attn.kv_cache
        virtual_engine = getattr(forward_context, "virtual_engine", None)
        if virtual_engine is not None and isinstance(compress_kv_cache, (list, tuple)):
            compress_kv_cache = compress_kv_cache[virtual_engine]
        # Patch 71: a V4.1 shared compressed layer (v41_kv_source_layer_name
        # set, assigned only to non-kv-source layers) must ALWAYS read its kv
        # source's compressed cache. With the generic kv-sharing grouping off
        # (VLLM_V41_DISABLE_KV_SHARING=1) the runner still allocates the
        # shared layer a private compressed cache; that allocation is
        # non-empty, so the previous "resolve the source only when the own
        # binding is empty" fallback never fired, and every shared layer
        # attended an ALL-ZERO compressed stream (op-level capture, decode
        # step 963: L2 cmp page 105/128 live rows vs L3 same pages 0 live
        # rows, identical top-k). Short generations stayed correct because
        # the window + sink + the eight owning layers carry them; long
        # generations progressively lost compressed context and degenerated
        # into the fixed-vocabulary soup. The V4 baseline never sets
        # v41_kv_source_layer_name, so its path is unchanged.
        v41_shared_target = getattr(self.dsa_attn, "v41_kv_source_layer_name", None)
        if v41_shared_target is not None:
            static_ctx = getattr(self, "_vllm_static_forward_context", None)
            target_attn = static_ctx.get(v41_shared_target) if static_ctx is not None else None
            target_cache = getattr(target_attn, "kv_cache", None) if target_attn is not None else None
            if isinstance(target_cache, (list, tuple)) and len(target_cache) > 0:
                ve = virtual_engine if virtual_engine is not None else 0
                target_cache = target_cache[ve] if ve < len(target_cache) else target_cache[0]
            if target_cache is not None and torch.is_tensor(target_cache) and target_cache.numel() > 0:
                compress_kv_cache = target_cache
        elif compress_kv_cache is None or (
            torch.is_tensor(compress_kv_cache) and compress_kv_cache.numel() == 0
        ):
            # V4.1 fallback: the runner did not alias this shared layer onto
            # its kv source's cache (empty binding). Read the source layer's
            # cache directly so the compressed stream is never empty (an
            # absent cmp_kv with cmp_ratio>1 is rejected by SparseFlashMla)
            # nor a never-written private cache.
            target_name = getattr(self.dsa_attn, "v41_kv_source_layer_name", None) or getattr(
                self.dsa_attn, "kv_sharing_target_layer_name", None
            )
            if target_name:
                static_ctx = get_current_vllm_config().compilation_config.static_forward_context
                target_attn = static_ctx.get(target_name)
                target_cache = getattr(target_attn, "kv_cache", None) if target_attn is not None else None
                if virtual_engine is not None and isinstance(target_cache, (list, tuple)):
                    target_cache = target_cache[virtual_engine]
                if target_cache is not None and torch.is_tensor(target_cache) and target_cache.numel() > 0:
                    compress_kv_cache = target_cache
    if self.indexer is not None and getattr(self.indexer, "owns_k", True):
        indexer_compressor = getattr(self.indexer, "compressor", None)
        if indexer_compressor is not None and indexer_compressor.state_cache is not None:
            indexer_state_cache = indexer_compressor.state_cache.kv_cache
        if get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE):
            indexer_k_cache, indexer_scale_cache, indexer_full_cache = unfold_kvcache(self.indexer.k_cache.kv_cache)
        else:
            indexer_k_cache, indexer_scale_cache = unfold_kvcache(self.indexer.k_cache.kv_cache)

    if get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE):
        kv_cache = tuple(
            [
                unfold_kvcache(cache)
                for cache in (
                    compress_kv_cache,
                    swa_kv_cache,
                    state_cache,
                    indexer_state_cache,
                    indexer_k_cache,
                    indexer_scale_cache,
                    indexer_full_cache,
                )
            ]
        )
    else:
        kv_cache = tuple(
            [
                unfold_kvcache(cache)
                for cache in (
                    compress_kv_cache,
                    swa_kv_cache,
                    state_cache,
                    indexer_state_cache,
                    indexer_k_cache,
                    indexer_scale_cache,
                )
            ]
        )
    return kv_cache


def unfold_kvcache(kvcache):
    while isinstance(kvcache, list) and len(kvcache) == 1:
        kvcache = kvcache[0]
    return kvcache
