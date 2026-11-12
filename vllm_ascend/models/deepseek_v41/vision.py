# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 vision tower, aligner and image-span delimiters.

Ported from the official vision.py / image_processor.py. Structure (and the
resulting checkpoint weight names) must match exactly:

- ViT: ``patch_embed.proj`` (3*p*p -> dim, bias), 32 blocks of
  ``attn.wqkv``/``attn.wo`` (both with bias, no qk norm), ``mlp.w1``/``mlp.w2``
  (no bias), ``norm1``/``norm2`` (fp32 RMSNorm, eps 1e-6), final ``norm``.
- Full bidirectional attention over one image with 2D RoPE (theta 10000),
  rope dim = head_dim // 2.
- Aligner: unfold r x r neighborhoods (r = downsample_ratio) then
  ``w1``/``w2`` (both with bias), GELU between.
- Delimiter vectors ``image_start`` / ``image_end`` / ``image_newline`` live on
  the language model (see model.py), not here.

Token types (image_processor semantics): an image span is
``[IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]``
with every position carrying ``image_token_id`` in input_ids.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

# token types inside an image span (TEXT positions are -1 elsewhere)
TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)


@lru_cache(maxsize=8)
def get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class VisionRMSNorm(nn.Module):
    """fp32 RMSNorm as in the official vision tower (eps defaults to 1e-6)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class PatchEmbed(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        patch_size = args.patch_size if hasattr(args, "patch_size") else args.vision_patch_size
        hidden = args.hidden_size if hasattr(args, "hidden_size") else args.vision_dim
        self.proj = nn.Linear(3 * patch_size**2, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class VisionAttention(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        self.n_heads = args.num_attention_heads if hasattr(args, "num_attention_heads") else args.vision_n_heads
        dim = args.hidden_size if hasattr(args, "hidden_size") else args.vision_dim
        self.head_dim = dim // self.n_heads
        self.wqkv = nn.Linear(dim, 3 * dim)
        self.wo = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        q, k, v = (t.view(n, self.n_heads, self.head_dim) for t in self.wqkv(x).chunk(3, dim=-1))
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        return self.wo(o.transpose(0, 1).reshape(n, -1))


class VisionMLP(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        dim = args.hidden_size if hasattr(args, "hidden_size") else args.vision_dim
        inter = args.intermediate_size if hasattr(args, "intermediate_size") else args.vision_inter_dim
        self.w1 = nn.Linear(dim, 2 * inter, bias=False)
        self.w2 = nn.Linear(inter, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class VisionBlock(nn.Module):
    def __init__(self, args: Any):
        super().__init__()
        dim = args.hidden_size if hasattr(args, "hidden_size") else args.vision_dim
        self.norm1 = VisionRMSNorm(dim)
        self.attn = VisionAttention(args)
        self.norm2 = VisionRMSNorm(dim)
        self.mlp = VisionMLP(args)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """DeepSeek ViT: full bidirectional attention over one image with 2D RoPE."""

    def __init__(self, args: Any):
        super().__init__()
        n_layers = args.num_hidden_layers if hasattr(args, "num_hidden_layers") else args.vision_n_layers
        n_heads = args.num_attention_heads if hasattr(args, "num_attention_heads") else args.vision_n_heads
        dim = args.hidden_size if hasattr(args, "hidden_size") else args.vision_dim
        self.rope_dim = dim // n_heads // 2
        self.rope_theta = args.rope_theta if hasattr(args, "rope_theta") else args.vision_rope_theta
        self.patch_embed = PatchEmbed(args)
        self.blocks = nn.ModuleList([VisionBlock(args) for _ in range(n_layers)])
        self.norm = VisionRMSNorm(dim)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        x = self.patch_embed(patches)
        # lru_cache builds the tables on CPU; move to the input device (the
        # official code relies on a global default device instead).
        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        cos = cos.to(patches.device)
        sin = sin.to(patches.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class Aligner(nn.Module):
    def __init__(self, args: Any, text_dim: int | None = None):
        super().__init__()
        self.downsample_ratio = (
            args.downsample_ratio if hasattr(args, "downsample_ratio") else args.vision_downsample_ratio
        )
        vision_dim = args.hidden_size if hasattr(args, "hidden_size") else args.vision_dim
        dim = text_dim if text_dim is not None else args.dim
        in_dim = vision_dim * self.downsample_ratio**2
        self.w1 = nn.Linear(in_dim, dim)
        self.w2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def image_token_types(n_llm_h: int, n_llm_w: int) -> torch.Tensor:
    """Default layout: the aligner grid in reading order, one IMAGE_NEW_LINE per row."""
    types = [IMAGE_START]
    types += ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
    types.append(IMAGE_END)
    return torch.tensor(types, dtype=torch.int64)


__all__ = [
    "TEXT",
    "IMAGE_START",
    "IMAGE",
    "IMAGE_NEW_LINE",
    "IMAGE_END",
    "Aligner",
    "ViT",
    "image_token_types",
    "num_image_tokens",
]
