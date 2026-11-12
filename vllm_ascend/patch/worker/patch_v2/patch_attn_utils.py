import vllm
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache

from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.patch.worker.patch_bind_kv_cache import bind_kv_cache
from vllm_ascend.worker.v2.attn_utils import (
    _allocate_kv_cache,
    _reshape_kv_cache_v2,
    get_kv_cache_spec,
)


def _get_ascend_sfa_indexer_backend(_self):
    return AscendSFAIndexerBackend


DeepseekV32IndexerCache.get_attn_backend = _get_ascend_sfa_indexer_backend
vllm.v1.worker.gpu.attn_utils._allocate_kv_cache = _allocate_kv_cache
vllm.v1.worker.gpu.attn_utils._reshape_kv_cache = _reshape_kv_cache_v2
vllm.v1.worker.gpu.attn_utils.bind_kv_cache = bind_kv_cache
vllm.v1.worker.gpu.model_runner.get_kv_cache_spec = get_kv_cache_spec


def _get_shared_kv_cache_layers_ascend(vllm_config):
    """V4.1 compat: DSAAttention implements AttentionLayerBase, not the core
    Attention class the upstream filter uses; include both so shared
    compressed-KV layers resolve their target cache."""
    from vllm.config.vllm import get_layers_from_vllm_config
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

    layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
    shared = {}
    for name, module in layers.items():
        tgt = getattr(module, "kv_sharing_target_layer_name", None)
        if tgt:
            shared[name] = tgt
    # keep anything the upstream filter already found
    from vllm.model_executor.layers.attention import Attention
    for name, module in get_layers_from_vllm_config(vllm_config, Attention).items():
        tgt = getattr(module, "kv_sharing_target_layer_name", None)
        if tgt:
            shared.setdefault(name, tgt)
    return shared


vllm.v1.worker.gpu.attn_utils.get_shared_kv_cache_layers = _get_shared_kv_cache_layers_ascend
