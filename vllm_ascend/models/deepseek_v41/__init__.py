# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 (DeepseekV41ForCausalLM) model implementation for Ascend.

Modules:
- config: re-export of the flattened DeepseekV41Config.
- model: AscendDeepseekV41ForCausalLM / decoder layers / MoE with bias_vl.
- engram: n-gram hash lookup tables added into the residual stream.
- indexer: V4.1 lightning indexer (weights_proj scoring, candidate blocks).
- vision: ViT + Aligner + image span delimiters.
"""
