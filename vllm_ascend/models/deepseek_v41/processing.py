# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 multimodal input processing (image path).

Ported from the official ``image_processor.py``: an image becomes an
``n_vit_h x n_vit_w`` patch grid for the ViT and an ``n_llm_h x n_llm_w``
token grid after the 3x3 aligner downsample, which the LLM sees as

    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]

Every position of that span carries ``image_token_id`` (129264) in
``input_ids``; only the token type (kept by the placeholder machinery as
``PlaceholderRange`` offsets) tells the positions apart.

Data flow (vLLM 0.27 multimodal contract):

- The chat endpoint replaces each ``image_url`` content block with the
  placeholder string ``<｜deepseek_image｜>`` (``get_placeholder_str`` on the
  model class), which the tokenizer encodes as the single token
  ``image_token_id``.
- This processor tokenizes the prompt text as-is and computes, per image,
  the ViT patch tensor plus its grid; the base class expands each
  placeholder token into its image span via ``_get_prompt_updates``
  (``_hf_processor_applies_updates`` returns False -- there is no HF
  processor for this model).
- The mm fields ``image_patches`` (flat, split via ``image_grids``) reach the
  model as kwargs of ``embed_multimodal``, which runs ViT+aligner and returns
  one embedding per image span (delimiters included); the runner scatters
  them over the placeholder positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image as PILImage
from PIL import ImageOps
from transformers import BatchFeature

from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import ImageProcessorItems, ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)

# Official encoding.IMAGE_PLACEHOLDER; tokenizes to the single token
# config.image_token_id (cross-checked at serve time).
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"

from vllm_ascend.models.deepseek_v41.vision import num_image_tokens  # noqa: E402


@dataclass(frozen=True)
class VisionParams:
    """Flat view of the checkpoint ``vision_config`` (+ ``image_token_id``)."""

    patch_size: int
    downsample_ratio: int
    min_pixels: int
    max_n_token: int
    max_wh_ratio: float | None
    image_token_id: int


def vision_params_from_config(hf_config: Any) -> VisionParams:
    vision = dict(getattr(hf_config, "vision_config", None) or {})
    return VisionParams(
        patch_size=int(vision.get("patch_size", 14)),
        downsample_ratio=int(vision.get("downsample_ratio", 3)),
        min_pixels=int(vision.get("min_pixels", 295936)),
        max_n_token=int(vision.get("max_image_tokens", 1024)),
        max_wh_ratio=vision.get("max_wh_ratio", None),
        image_token_id=int(getattr(hf_config, "image_token_id", 129264)),
    )


# ---------------------------------------------------------------------------
# Grid planning -- verbatim port of the official image_processor.py math
# (pure functions; keep bit-identical arithmetic).
# ---------------------------------------------------------------------------


def llm_grid(best_height: int, best_width: int, patch_size: int, downsample_ratio: int):
    """Token grid the aligner produces from a patch grid of this pixel size."""
    return math.ceil((best_height // patch_size) / downsample_ratio), math.ceil(
        (best_width // patch_size) / downsample_ratio
    )


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    """Largest aspect-preserving pixel size whose token grid still fits in max_n_token."""
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:  # very tall: collapse to a single column
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:  # very wide: collapse to a single row
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    return math.floor(height * beta / patch_size) * patch_size, math.floor(width * beta / patch_size) * patch_size


def safe_resize(height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token):
    """Shrink the pixel size until the image costs at most max_n_token LLM tokens."""
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token)
        n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


def plan_image_grid(width: int, height: int, params: VisionParams):
    """Resize plan for an image of the given original size; a pure function of its arguments."""
    p = params.patch_size
    if params.max_wh_ratio is not None and width > height * params.max_wh_ratio:
        width = height * params.max_wh_ratio
    if 0 < width * height < params.min_pixels:
        ratio = (params.min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    return safe_resize(height, width, best_height, best_width, p, params.downsample_ratio, params.max_n_token)


def image_to_patches(
    image: PILImage.Image, params: VisionParams
) -> tuple[torch.Tensor, int, int, int, int]:
    """Official ``load_image``: PIL image -> (patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w).

    patches: [n_vit_h * n_vit_w, 3, p, p] bf16, normalized to [-1, 1].
    """
    p = params.patch_size
    image = image.convert("RGB")
    n_llm_h, n_llm_w, best_height, best_width = plan_image_grid(image.width, image.height, params)
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if params.max_wh_ratio is not None and image.width >= params.max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    patches = x.reshape(3, n_vit_h, p, n_vit_w, p).permute(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3, p, p)
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def _to_pil(item: Any) -> PILImage.Image:
    if isinstance(item, PILImage.Image):
        return item
    if isinstance(item, np.ndarray):
        return PILImage.fromarray(item)
    if isinstance(item, torch.Tensor):
        return PILImage.fromarray(item.detach().cpu().numpy())
    raise TypeError(f"Unsupported image item type for the V4.1 processor: {type(item)}")


def max_square_image_size(params: VisionParams) -> ImageSize:
    """Largest square pixel size (multiple of patch*downsample_ratio) whose
    image span fits in ``max_n_token``; used for profiling dummy images."""
    h = max(1, int(math.sqrt(params.max_n_token - 2)))
    while h > 1 and h * (h + 1) + 2 > params.max_n_token:
        h -= 1
    side = h * params.patch_size * params.downsample_ratio
    return ImageSize(width=side, height=side)


# ---------------------------------------------------------------------------
# vLLM processor plumbing
# ---------------------------------------------------------------------------


class DeepseekV41ProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def vision_params(self) -> VisionParams:
        return vision_params_from_config(self.get_hf_config())

    def get_image_size_with_most_features(self) -> ImageSize:
        return max_square_image_size(self.vision_params())


class DeepseekV41DummyInputsBuilder(BaseDummyInputsBuilder[DeepseekV41ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        return IMAGE_PLACEHOLDER * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        max_image_size = self.info.get_image_size_with_most_features()
        image_overrides = mm_options.get("image")
        return {
            "image": self._get_dummy_images(
                width=max_image_size.width,
                height=max_image_size.height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class DeepseekV41MultiModalProcessor(BaseMultiModalProcessor[DeepseekV41ProcessingInfo]):
    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        # No HF processor exists for this model: we tokenize the prompt text
        # as-is and let the base class expand the placeholder tokens via
        # _get_prompt_updates.
        return False

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        params = self.info.vision_params()

        # NOTE: the deepseek tokenizer never adds extra special tokens on
        # encode (the chat template already contains <｜begin▁of▁sentence｜>),
        # so this matches the official tokenizer.encode(prompt) exactly.
        add_special_tokens = tok_kwargs.get("add_special_tokens", True)
        if isinstance(add_special_tokens, torch.Tensor):
            add_special_tokens = bool(add_special_tokens)
        ids = tokenizer.encode(prompt, add_special_tokens=bool(add_special_tokens))
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        out: dict[str, Any] = {"input_ids": [ids]}

        # mm_data key is plural ("images") -- see ProcessorBatchItems.get_processor_data.
        images = mm_data.get("images") or mm_data.get("image") or []
        if len(images) > 0:
            patches_list: list[torch.Tensor] = []
            grids: list[tuple[int, int]] = []
            sizes: list[int] = []
            for item in images:
                image = _to_pil(item)
                patches, n_vit_h, n_vit_w, _n_llm_h, _n_llm_w = image_to_patches(image, params)
                patches_list.append(patches)
                grids.append((n_vit_h, n_vit_w))
                sizes.append(patches.shape[0])
            out["image_patches"] = torch.cat(patches_list, dim=0)
            out["image_grids"] = torch.tensor(grids, dtype=torch.int64)
            out["image_patch_sizes"] = torch.tensor(sizes, dtype=torch.int64)
        return BatchFeature(data=out)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        sizes = hf_inputs.get("image_patch_sizes", torch.empty(0, dtype=torch.int64))
        return dict(
            image_patches=MultiModalFieldConfig.flat_from_sizes("image", sizes),
            image_grids=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        params = self.info.vision_params()
        image_token_id = params.image_token_id

        def get_replacement(item_idx: int) -> list[int]:
            images = mm_items.get_items("image", (ImageProcessorItems,))
            image_size = images.get_image_size(item_idx)
            n_llm_h, n_llm_w, _, _ = plan_image_grid(image_size.width, image_size.height, params)
            return [image_token_id] * num_image_tokens(n_llm_h, n_llm_w)

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            )
        ]


__all__ = [
    "IMAGE_PLACEHOLDER",
    "VisionParams",
    "vision_params_from_config",
    "llm_grid",
    "solve_resize_ratio",
    "safe_resize",
    "plan_image_grid",
    "image_to_patches",
    "max_square_image_size",
    "DeepseekV41ProcessingInfo",
    "DeepseekV41DummyInputsBuilder",
    "DeepseekV41MultiModalProcessor",
]
