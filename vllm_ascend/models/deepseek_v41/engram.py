# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engram: n-gram hash lookups written into the hyper-connection residual stream.

Ported from the official DeepSeek-V4.1 reference (engram.py). Semantics that
must not drift:

- ``EngramLayout`` draws one prime per (layer, n-gram size, head) bucket range,
  in order, never reusing a prime. Primes start above ``engram_vocab_size - 1``.
- ``compute_hash_multipliers`` uses ``numpy.random.default_rng(10007 * layer_id)``
  and derives odd multipliers bounded so ``token * multiplier`` cannot overflow
  int64. The RNG stream is part of the checkpoint contract: any change rehashes
  the whole table.
- The hash itself is a running XOR of multiplied ids; after step ``i`` the
  value is the hash of the (i+1)-gram, reduced into the (i)-th ngram bucket
  range and shifted by that range's offset.
- The fp8 table rows are dequantized on lookup (fp8 * ue8m0 per-32 block).

The tables (~98 GB each at the released scale) are NOT module parameters:
they stay CPU-side as safetensors mmaps and only the selected rows are moved
to the device per step (first version is synchronous; overlap is a later
item). Multiple DP ranks sharing one checkpoint therefore share the same
physical pages.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

try:  # safetensors is a hard vllm dependency; keep the import lazy-safe anyway
    from safetensors import safe_open
except ImportError:  # pragma: no cover
    safe_open = None

FP8_BLOCK_SIZE = 32


def _is_prime(n: int) -> bool:
    """Exact primality by trial division (deterministic, matches sympy.isprime)."""
    if n < 2:
        return False
    for p in (2, 3):
        if n % p == 0:
            return n == p
    if n % 6 not in (1, 5):
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """The smallest prime above ``start`` that has not been handed out yet."""
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def compute_hash_multipliers(
    layer_ids: tuple[int, ...],
    max_ngram_size: int,
    tokenizer_vocab_size: int,
) -> torch.Tensor:
    """One multiplier per (layer, lookback), from a per-layer RNG so layers hash differently.

    Kept odd, and bounded so that ``token_id * multiplier`` cannot overflow int64.
    Bit-identical to the official implementation (same numpy RNG stream).
    """
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space where tokens that normalize alike collapse together.

    N-grams are hashed over these compressed ids, so " The", "the" and "THE"
    all hash the same way. Returns the lookup plus the size of the compressed
    vocab -- and that size matters beyond bounds checking, because every hash
    multiplier is derived from it.
    """
    from tokenizers import Regex, normalizers

    # a private-use char, so a token that is exactly one space survives Strip()
    # instead of collapsing to the empty string and merging with unrelated
    # tokens
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    # the raw Rust tokenizer, matching what training decodes with
    # (no clean_up_tokenization_spaces)
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # a partial UTF-8 byte token: nothing to normalize, so key it by
            # its raw form
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


@dataclass(frozen=True)
class EngramLayout:
    """Bucket layout of the n-gram hash tables.

    A position is hashed as ``max_ngram_size - 1`` n-grams (2-gram ..
    max_ngram_size-gram), each split over ``n_heads`` heads. Every (n-gram
    size, head) pair owns its own prime-sized bucket range in the layer's
    table; the primes are drawn in order and never reused, which keeps the
    ranges disjoint.
    """

    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]  # table rows, per engram layer
    primes: tuple[tuple[tuple[int, ...], ...], ...]  # [layer][n-gram size][head]
    n_heads: int
    head_dim: int

    @classmethod
    def from_args(cls, args: Any) -> EngramLayout | None:
        layer_ids = tuple(args.engram_layer_ids)
        if not layer_ids:
            return None
        max_ngram_size, n_heads = args.engram_max_ngram_size, args.engram_n_heads
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram_size - 1):
                sizes, current = [], args.engram_vocab_size - 1
                for _ in range(n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=tuple(args.engram_num_embeddings),
            primes=tuple(primes),
            n_heads=n_heads,
            head_dim=args.engram_head_dim,
        )

    @property
    def n_hash_cols(self) -> int:
        return (self.max_ngram_size - 1) * self.n_heads


def compute_ngram_hashes(
    token_rows: torch.Tensor,
    positions: torch.Tensor,
    history: torch.Tensor,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
    pad_id: int,
    dead_id: int = -1,
) -> torch.Tensor:
    """Pure hash math, identical to the official NgramHashState.forward.

    ``token_rows``: [B, L] compressed ids for the current chunk (DEAD where
    the token takes no part in an n-gram; used for the output shape only).
    ``history``: [B, max_pos] cache holding compressed ids of every position
    up to and including the current chunk. ``positions``: [B, L] absolute
    position of each current token. Returns [B, L, n_hash_cols] table row ids.
    """
    batch, seqlen = token_rows.shape
    tokens_list, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
    for shift in range(multipliers.shape[-1]):  # max_ngram_size
        source = history[:batch].gather(1, (positions - shift).clamp_min(0))
        blocked = blocked | (positions < shift) | (source == dead_id)
        tokens_list.append(torch.where(blocked, pad_id, source))
    tokens = torch.stack(tokens_list, dim=-1)  # [B, L, max_ngram_size]

    # XOR the multiplied ids together one lookback at a time, so the running
    # value after step i is the hash of the (i+1)-gram; each lands in its own
    # prime-sized bucket range
    products = tokens.unsqueeze(2) * multipliers  # [B, L, n_engram_layers, max_ngram_size]
    rolling, hashes = products[..., 0], []
    for i in range(1, multipliers.shape[-1]):
        rolling = torch.bitwise_xor(rolling, products[..., i])
        hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1])
    return torch.cat(hashes, dim=-1) + offsets


class NgramHashState:
    """Per-request token history producing [B, L, n_layers, n_hash_cols] hash ids.

    vLLM-side batch semantics: state is a CPU dict keyed by request id; the
    prefill chunk builds the full history and every decode step appends one
    token. Image positions are recorded as DEAD so an n-gram never spans one.
    """

    DEAD = -1

    def __init__(self, args: Any, layout: EngramLayout, tokenizer=None):
        self.layout = layout
        self.pad_token_id = args.engram_pad_token_id
        self.max_seq_len = getattr(args, "max_position_embeddings", 0) or 1 << 20
        if tokenizer is not None:
            token_map, vocab_size = build_compressed_token_map(tokenizer)
            expected = args.engram_compressed_vocab_size
            if vocab_size != expected:
                raise ValueError(
                    f"compressed tokenizer vocab {vocab_size} != config "
                    f"engram_compressed_vocab_size {expected}; every hash "
                    "multiplier derives from this size, refusing to rehash"
                )
            self.pad_id = token_map[args.engram_pad_token_id]
            self.token_map = torch.tensor(token_map, dtype=torch.int64, device="cpu")
            self.compressed_vocab_size = vocab_size
        else:
            # Deferred tokenizer binding (Pass B wires the serve-time
            # tokenizer); hashing stays unusable until bound.
            self.pad_id = None
            self.token_map = None
            self.compressed_vocab_size = args.engram_compressed_vocab_size
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in layout.primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        multipliers = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, self.compressed_vocab_size)
        self.primes = torch.tensor(layout.primes, device="cpu")  # [n_layers, ngram-1, heads]
        self.offsets = torch.tensor(np.array(offsets), device="cpu")  # [n_layers, n_hash_cols]
        # hash math stays on CPU (histories are CPU rows); compute_hash_
        # multipliers builds under the caller's default device, which inside a
        # vLLM NPU worker is the NPU.
        self.multipliers = multipliers.cpu() if multipliers.device.type != "cpu" else multipliers
        self._history: dict[str, torch.Tensor] = {}
        self._lengths: dict[str, int] = {}

    def bind_tokenizer(self, tokenizer) -> None:
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        # NOTE: layout/multipliers were already derived from the config's
        # compressed vocab; a mismatching tokenizer is a hard error.
        if vocab_size != self.compressed_vocab_size:
            raise ValueError(
                f"compressed tokenizer vocab {vocab_size} != "
                f"{self.compressed_vocab_size}; refusing to rehash the engram tables"
            )
        self.token_map = torch.tensor(token_map, dtype=torch.int64, device="cpu")
        self.pad_id = token_map[self.pad_token_id]

    def _history_row(self, req_id: str) -> torch.Tensor:
        row = self._history.get(req_id)
        if row is None:
            row = torch.full((self.max_seq_len,), self.DEAD, dtype=torch.int64, device="cpu")
            self._history[req_id] = row
            self._lengths[req_id] = 0
        return row

    def seek(self, req_id: str, start_pos: int) -> None:
        """Align a history row to where the next chunk starts.

        The row is keyed by the request's first KV block id, which the
        allocator can reuse for a DIFFERENT request once the original blocks
        are freed (or share, via prefix caching, with a request whose prompt
        only agrees on a prefix). Appending blindly then reads a stale
        history as if it were this request's tokens. Alignment rules:
          - length == start_pos: plain continuation (decode steps).
          - start_pos == 0: fresh request on a reused id: restart from 0.
          - length > start_pos: prefix-cache hit: the row prefix [0,
            start_pos) IS the shared content (identical blocks); truncate.
          - length < start_pos: history gap (evicted/reused before we saw
            these tokens): DEAD-fill the gap so no foreign tokens leak in.
        """
        length = self._lengths.get(req_id, 0)
        if length == start_pos:
            return
        row = self._history_row(req_id)
        if start_pos == 0:
            self._lengths[req_id] = 0
        elif length > start_pos:
            self._lengths[req_id] = start_pos
        else:
            row[length:start_pos] = self.DEAD
            self._lengths[req_id] = start_pos

    def extend(self, req_id: str, input_ids: torch.Tensor, is_image: torch.Tensor | None) -> None:
        """Record a chunk of real token ids; ``is_image`` True positions become DEAD."""
        row = self._history_row(req_id)
        ids = input_ids.tolist() if input_ids.dim() == 1 else input_ids[0].tolist()
        if is_image is not None:
            mask = is_image.tolist() if is_image.dim() == 1 else is_image[0].tolist()
        else:
            mask = [False] * len(ids)
        start = self._lengths[req_id]
        for offset, (token, img) in enumerate(zip(ids, mask)):
            row[start + offset] = self.DEAD if img else int(self.token_map[token].item())
        self._lengths[req_id] = start + len(ids)
        # bound the number of live history rows (each is max_seq_len int64)
        if not hasattr(self, "_lru"):
            self._lru = OrderedDict()
        self._lru[req_id] = None
        self._lru.move_to_end(req_id)
        while len(self._lru) > 256:
            victim, _ = next(iter(self._lru.items()))
            self._lru.pop(victim)
            self._history.pop(victim, None)
            self._lengths.pop(victim, None)

    def hashes(
        self,
        req_ids: list[str],
        start_pos: list[int],
        num_tokens: list[int],
    ) -> torch.Tensor:
        """Hash ids for a batch of requests: [B, L, n_layers, n_hash_cols].

        All requests in the batch must share the same chunk length (vLLM
        prefill chunks and uniform decode steps both satisfy this).
        """
        rows, positions, histories = [], [], []
        for req_id, start, count in zip(req_ids, start_pos, num_tokens):
            row = self._history[req_id]
            rows.append(row[start : start + count])
            positions.append(torch.arange(start, start + count, dtype=torch.int64, device="cpu"))
            histories.append(self._history[req_id])
        token_rows = torch.stack(rows)  # [B, L]
        position_rows = torch.stack(positions)
        pad = self.pad_id if self.pad_id is not None else 0
        return compute_ngram_hashes(
            token_rows,
            position_rows,
            torch.stack(histories),
            self.multipliers,
            self.primes,
            self.offsets,
            pad,
            self.DEAD,
        )


class EngramEmbeddingTable(nn.Module):
    """The n-gram hash table. NOT a parameter holder: rows live in a CPU mmap.

    ``bind`` records the safetensors file and tensor names; ``lookup`` gathers
    rows on demand, dequantizing fp8*ue8m0 blocks, and ships only the selected
    rows to the device. Unbound lookups raise: the loader skips these tensors
    (98 GB at the released scale must never be materialized through the
    normal device loader).
    """

    def __init__(self, num_embeddings: int, head_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.head_dim = head_dim
        self.block_size = FP8_BLOCK_SIZE
        self._file_path: str | None = None
        self._weight_name: str | None = None
        self._scale_name: str | None = None
        self._weight_slice = None
        self._scale_slice = None

    @property
    def bound(self) -> bool:
        return self._weight_slice is not None

    def bind(self, file_path: str, weight_name: str, scale_name: str) -> None:
        if safe_open is None:  # pragma: no cover
            raise RuntimeError("safetensors is required to mmap engram tables")
        with safe_open(file_path, framework="pt", device="cpu") as f:
            self._weight_slice = f.get_slice(weight_name)
            self._scale_slice = f.get_slice(scale_name)
            shape = tuple(self._weight_slice.get_shape())
        if shape != (self.num_embeddings, self.head_dim):
            raise ValueError(
                f"engram table {weight_name} has shape {shape}, expected "
                f"{(self.num_embeddings, self.head_dim)}"
            )
        self._file_path = file_path
        self._weight_name = weight_name
        self._scale_name = scale_name

    @staticmethod
    def _dequantize(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # ue8m0 scale byte -> 2 ** (byte - 127), one scale per 32 columns
        exponent = scale.view(torch.uint8).to(torch.float32) if scale.dtype not in (torch.float32, torch.float64) else scale
        if scale.dtype not in (torch.float32, torch.float64):
            exponent = exponent - 127.0
            factor = torch.pow(2.0, exponent)
        else:
            factor = scale
        values = weight.to(torch.float32).unflatten(-1, (-1, FP8_BLOCK_SIZE))
        return (values * factor.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)

    def lookup(self, row_ids: torch.Tensor, device: torch.device | str = "npu") -> torch.Tensor:
        """Gather table rows for arbitrary ids: returns [*, head_dim] bf16 on ``device``.

        ``row_ids`` must already be clamped to [0, num_embeddings) with invalid
        positions mapped to a dedicated zero row or masked by the caller.
        """
        if not self.bound:
            raise RuntimeError(
                "engram table is not bound to a safetensors mmap yet; call "
                "bind() (see AscendDeepseekV41ForCausalLM.bind_engram_tables)"
            )
        flat = row_ids.reshape(-1).to(torch.int64).cpu()
        unique, inverse = torch.unique(flat, return_inverse=True)
        # merge consecutive ids into contiguous slices (PySafeSlice supports
        # range slicing only)
        runs: list[tuple[int, int]] = []
        start = prev = int(unique[0].item())
        for value in unique[1:].tolist():
            if value == prev + 1:
                prev = value
                continue
            runs.append((start, prev + 1))
            start = prev = value
        runs.append((start, prev + 1))
        weight_parts, scale_parts = [], []
        for run_start, run_end in runs:
            weight_parts.append(self._weight_slice[run_start:run_end].to(torch.float8_e4m3fn))
            scale_parts.append(self._scale_slice[run_start:run_end].to(torch.float8_e8m0fnu))
        weights = torch.cat(weight_parts, dim=0)
        scales = torch.cat(scale_parts, dim=0)
        rows = self._dequantize(weights, scales)
        gathered = rows.index_select(0, inverse)
        return gathered.to(device).reshape(*row_ids.shape, self.head_dim)


class Engram(nn.Module):
    """Writes an n-gram lookup into the residual stream, gated by how well it matches that stream.

    The hash ids fetch ``n_hash_cols`` rows; ``wkv`` turns them into one key
    per hc copy plus a shared value. The gate is a normalized dot product of
    stream against key.
    """

    def __init__(
        self,
        config: Any,
        layer_id: int,
        layout: EngramLayout,
        quant_config: Any | None = None,
        prefix: str = "",
    ):
        super().__init__()
        # Imported here to keep this module importable without the NPU stack.
        from vllm.model_executor.layers.linear import ReplicatedLinear

        self.layer_id = layer_id
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.dim = config.hidden_size
        self.hc_mult = config.hc_mult
        self.clamp_value = 1e-6
        n_hash_cols = layout.n_hash_cols

        self.embed = EngramEmbeddingTable(layout.num_embeddings[self.layer_hash_index], layout.head_dim)
        self.wkv = ReplicatedLinear(
            n_hash_cols * layout.head_dim,
            self.dim * (self.hc_mult + 1),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wkv",
            return_bias=False,
        )
        self.eps = config.rms_norm_eps
        # only ever used as a product; fp32 holds the checkpoint exactly
        self.q_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim, dtype=torch.float32))
        self.k_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x: [..., hc_mult, dim]; hash_ids: [..., n_hash_cols]; token_mask: [...],
        False shuts the gate so those positions pass through untouched."""
        if not self.embed.bound:
            raise NotImplementedError(
                "engram table mmap is not bound; hash-state integration is "
                "deferred (bind_engram_tables + NgramHashState wiring)"
            )
        embedded = self.embed.lookup(hash_ids, device=x.device)  # [..., n_hash_cols, head_dim]
        return self.apply_embedded(x, embedded, token_mask=token_mask)

    def apply_embedded(
        self,
        x: torch.Tensor,
        embedded: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The device-side remainder of forward over PRE-GATHERED table rows.

        Only the CPU-mmap row gather is host work; everything here is device
        math on the official definitions. The graph decode path stages the
        gathered rows in a persistent buffer each step (host gather runs in
        the eager prepare hook — it cannot run inside the captured graph),
        then calls this; eager forward is the same math via lookup().
        """
        lead_shape = x.shape[:-2]
        wkv_out = self.wkv(embedded.reshape(*lead_shape, -1))
        # vLLM Linear.forward returns (output, bias); some quant methods on
        # this stack return the bare output tensor.
        kv = wkv_out[0] if isinstance(wkv_out, tuple) else wkv_out
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()  # only ever used as a product
        h, eps = x.float(), self.eps
        # normalized per (token, hc copy) over `dim`, NOT jointly over the copies
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        # signed sqrt before the sigmoid, matching the training kernel.
        # copysign(x, dot) == select(dot >= 0, x, -x) numerically (the -0 vs
        # +0 distinction never survives sigmoid); the plain select keeps this
        # on NPU kernels — aten::copysign falls back to CPU (an H2D copy that
        # breaks graph capture).
        s = dot.abs().clamp_min(self.clamp_value).sqrt()
        gate = torch.sigmoid(torch.where(dot >= 0, s, -s))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


def engram_weight_names(layer_id: int) -> tuple[str, ...]:
    """Checkpoint-side names owned by one engram layer (for the loader skip list)."""
    return (
        f"layers.{layer_id}.engram.embed.weight",
        f"layers.{layer_id}.engram.embed.scale",
    )


__all__ = [
    "EngramLayout",
    "Engram",
    "EngramEmbeddingTable",
    "NgramHashState",
    "build_compressed_token_map",
    "compute_hash_multipliers",
    "compute_ngram_hashes",
    "find_next_prime",
    "engram_weight_names",
]
