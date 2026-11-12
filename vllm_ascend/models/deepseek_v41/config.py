# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 config shim for vllm-ascend.

The canonical flattened config lives in vllm core
(``vllm.transformers_utils.configs.deepseek_v41.DeepseekV41Config``) so the
``_CONFIG_REGISTRY`` entry can resolve it; this module re-exports it for the
familiar ``vllm_ascend.models.deepseek_v41.config`` import path.
"""

from vllm.transformers_utils.configs.deepseek_v41 import (  # noqa: F401
    DeepseekV41Config,
    _flatten_v41_text_config,
)
