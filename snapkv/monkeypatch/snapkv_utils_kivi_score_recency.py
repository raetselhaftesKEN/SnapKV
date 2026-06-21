import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn


# Copied from transformers.models.llama.modeling_llama.repeat_kv
# This helper is kept for compatibility with old SnapKV code.
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@dataclass
class KIVIQuantizedTensor:
    """
    A small, pure-PyTorch portable representation of KIVI-style group-wise affine quantization.

    packed: uint8 tensor. For bits=2, one byte stores 4 values. For bits=4, one byte stores 2 values.
    scale/mn: fp16/fp32 tensors with shape orig_shape[:-1] + [num_groups].
    orig_shape: original unpadded shape before quantization along the last dimension.
    pad_len: number of padded elements appended to the last dimension before quantization/packing.
    """
    packed: torch.Tensor
    scale: torch.Tensor
    mn: torch.Tensor
    orig_shape: Tuple[int, ...]
    bits: int
    group_size: int
    pad_len: int

    @property
    def device(self):
        return self.packed.device

    def nbytes(self) -> int:
        total = 0
        for t in (self.packed, self.scale, self.mn):
            if t is not None:
                total += int(t.numel() * t.element_size())
        return total


class KIVIUniformQuantizer:
    """
    Portable KIVI-style quantizer.

    KIVI's paper/code use asymmetric group-wise min-max affine quantization:
        q = round((x - mn) / scale),  scale = (mx - mn) / (2**bits - 1)
        x_hat = q * scale + mn

    This implementation uses uint8 bit-packing instead of KIVI's int32 CUDA layout so it can be dropped
    into arbitrary PyTorch code without compiling custom CUDA. It is intended for correctness/ablation.
    Replace this class with KIVI's quant/new_pack.py + quant/matmul.py + quant/csrc/* for speed.
    """
    def __init__(self, bits: int = 2, group_size: int = 32, eps: float = 1e-6):
        if bits not in (2, 4, 8):
            raise ValueError(f"bits must be 2, 4, or 8, got {bits}")
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        self.bits = bits
        self.group_size = group_size
        self.eps = eps

    def _pack_uint8(self, q: torch.Tensor) -> torch.Tensor:
        q = q.to(torch.uint8).contiguous()
        if self.bits == 8:
            return q
        if self.bits == 4:
            if q.shape[-1] % 2 != 0:
                q = F.pad(q, (0, 1), value=0)
            q = q.view(*q.shape[:-1], q.shape[-1] // 2, 2)
            return (q[..., 0] | (q[..., 1] << 4)).contiguous()
        # bits == 2
        if q.shape[-1] % 4 != 0:
            q = F.pad(q, (0, 4 - q.shape[-1] % 4), value=0)
        q = q.view(*q.shape[:-1], q.shape[-1] // 4, 4)
        return (q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)).contiguous()

    def _unpack_uint8(self, packed: torch.Tensor, unpacked_last_dim: int) -> torch.Tensor:
        packed = packed.to(torch.uint8).contiguous()
        if self.bits == 8:
            q = packed
        elif self.bits == 4:
            q0 = packed & 0x0F
            q1 = (packed >> 4) & 0x0F
            q = torch.stack((q0, q1), dim=-1).flatten(-2)
        else:
            q0 = packed & 0x03
            q1 = (packed >> 2) & 0x03
            q2 = (packed >> 4) & 0x03
            q3 = (packed >> 6) & 0x03
            q = torch.stack((q0, q1, q2, q3), dim=-1).flatten(-2)
        return q[..., :unpacked_last_dim].contiguous()

    @torch.no_grad()
    def quantize_last_dim(self, x: torch.Tensor) -> KIVIQuantizedTensor:
        if x.numel() == 0:
            # Preserve the metadata for empty dropped-cache cases.
            return KIVIQuantizedTensor(
                packed=torch.empty((*x.shape[:-1], 0), device=x.device, dtype=torch.uint8),
                scale=torch.empty((*x.shape[:-1], 0), device=x.device, dtype=x.dtype),
                mn=torch.empty((*x.shape[:-1], 0), device=x.device, dtype=x.dtype),
                orig_shape=tuple(x.shape),
                bits=self.bits,
                group_size=self.group_size,
                pad_len=0,
            )
        orig_shape = tuple(x.shape)
        last = x.shape[-1]
        pad_len = (self.group_size - last % self.group_size) % self.group_size
        if pad_len:
            x_pad = F.pad(x, (0, pad_len), value=0.0)
        else:
            x_pad = x
        padded_last = x_pad.shape[-1]
        x_group = x_pad.reshape(*x_pad.shape[:-1], padded_last // self.group_size, self.group_size)
        mn = x_group.amin(dim=-1)
        mx = x_group.amax(dim=-1)
        scale = (mx - mn).clamp_min(self.eps) / float((1 << self.bits) - 1)
        q = torch.round((x_group - mn.unsqueeze(-1)) / scale.unsqueeze(-1))
        q = q.clamp_(0, (1 << self.bits) - 1).to(torch.uint8)
        q = q.reshape(*x_pad.shape[:-1], padded_last)
        packed = self._pack_uint8(q)
        return KIVIQuantizedTensor(
            packed=packed,
            scale=scale.to(x.dtype),
            mn=mn.to(x.dtype),
            orig_shape=orig_shape,
            bits=self.bits,
            group_size=self.group_size,
            pad_len=pad_len,
        )

    @torch.no_grad()
    def dequantize_last_dim(self, qt: KIVIQuantizedTensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if qt.bits != self.bits or qt.group_size != self.group_size:
            raise ValueError(
                f"Quantized tensor metadata mismatch: tensor has bits={qt.bits}, group_size={qt.group_size}; "
                f"quantizer has bits={self.bits}, group_size={self.group_size}"
            )
        padded_last = qt.orig_shape[-1] + qt.pad_len
        if padded_last == 0:
            return torch.empty(qt.orig_shape, device=qt.device, dtype=dtype or qt.scale.dtype)
        q = self._unpack_uint8(qt.packed, padded_last).to(qt.scale.dtype)
        q_group = q.reshape(*qt.orig_shape[:-1], padded_last // qt.group_size, qt.group_size)
        x = q_group * qt.scale.unsqueeze(-1) + qt.mn.unsqueeze(-1)
        x = x.reshape(*qt.orig_shape[:-1], padded_last)[..., : qt.orig_shape[-1]]
        return x.to(dtype or qt.scale.dtype).contiguous()

    @torch.no_grad()
    def quantize_key_per_channel(self, key_states: torch.Tensor) -> KIVIQuantizedTensor:
        # key_states: [B, H, T, D]. Per-channel means group along token dim for every D channel.
        return self.quantize_last_dim(key_states.transpose(-1, -2).contiguous())  # [B,H,D,T]

    @torch.no_grad()
    def dequantize_key_per_channel(self, qt: Optional[KIVIQuantizedTensor], dtype: torch.dtype) -> Optional[torch.Tensor]:
        if qt is None:
            return None
        x = self.dequantize_last_dim(qt, dtype=dtype)  # [B,H,D,T]
        return x.transpose(-1, -2).contiguous()        # [B,H,T,D]



    @torch.no_grad()
    def dequantize_value_per_token_range(
        self,
        qt: Optional[KIVIQuantizedTensor],
        start: int,
        end: int,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Dequantize only value tokens [start:end]. Value qt shape is [B,H,T,D]."""
        if qt is None:
            return None
        if end <= start:
            b, h, _, d = qt.orig_shape
            return torch.empty((b, h, 0, d), device=qt.device, dtype=dtype)
        b, h, t, d = qt.orig_shape
        start = max(0, min(start, t))
        end = max(start, min(end, t))
        sub = KIVIQuantizedTensor(
            packed=qt.packed[:, :, start:end, :].contiguous(),
            scale=qt.scale[:, :, start:end, :].contiguous(),
            mn=qt.mn[:, :, start:end, :].contiguous(),
            orig_shape=(b, h, end - start, d),
            bits=qt.bits,
            group_size=qt.group_size,
            pad_len=qt.pad_len,
        )
        return self.dequantize_last_dim(sub, dtype=dtype)

    @torch.no_grad()
    def dequantize_key_per_channel_range(
        self,
        qt: Optional[KIVIQuantizedTensor],
        start: int,
        end: int,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Dequantize only key tokens [start:end]. Key qt shape is [B,H,D,T] before transpose back."""
        if qt is None:
            return None
        b, h, d, t = qt.orig_shape
        start = max(0, min(start, t))
        end = max(start, min(end, t))
        if end <= start:
            return torch.empty((b, h, 0, d), device=qt.device, dtype=dtype)
        if qt.bits != self.bits or qt.group_size != self.group_size:
            raise ValueError(
                f"Quantized tensor metadata mismatch: tensor has bits={qt.bits}, group_size={qt.group_size}; "
                f"quantizer has bits={self.bits}, group_size={self.group_size}"
            )

        # Key was quantized along token dimension. Slice in quantization-group aligned units
        # to keep each group's scale/min valid, then crop back to [start:end].
        group = qt.group_size
        pack_factor = 1 if qt.bits == 8 else (8 // qt.bits)
        padded_t = t + qt.pad_len
        aligned_start = (start // group) * group
        aligned_end = min(((end + group - 1) // group) * group, padded_t)
        aligned_len = max(0, aligned_end - aligned_start)
        g0 = aligned_start // group
        g1 = aligned_end // group
        byte0 = aligned_start // pack_factor
        byte1 = (aligned_end + pack_factor - 1) // pack_factor

        packed = qt.packed[:, :, :, byte0:byte1].contiguous()
        scale = qt.scale[:, :, :, g0:g1].contiguous()
        mn = qt.mn[:, :, :, g0:g1].contiguous()
        q = self._unpack_uint8(packed, aligned_len).to(scale.dtype)
        q_group = q.reshape(b, h, d, aligned_len // group, group)
        x = q_group * scale.unsqueeze(-1) + mn.unsqueeze(-1)
        x = x.reshape(b, h, d, aligned_len)
        local0 = start - aligned_start
        local1 = local0 + (end - start)
        x = x[..., local0:local1]
        return x.transpose(-1, -2).to(dtype).contiguous()
    @torch.no_grad()
    def quantize_value_per_token(self, value_states: torch.Tensor) -> KIVIQuantizedTensor:
        # value_states: [B, H, T, D]. Per-token means group along channel/head_dim dim.
        return self.quantize_last_dim(value_states.contiguous())

    @torch.no_grad()
    def dequantize_value_per_token(self, qt: Optional[KIVIQuantizedTensor], dtype: torch.dtype) -> Optional[torch.Tensor]:
        if qt is None:
            return None
        return self.dequantize_last_dim(qt, dtype=dtype)


@dataclass
class SnapKVKIVIDroppedCache:
    key_q: Optional[KIVIQuantizedTensor]
    value_q: Optional[KIVIQuantizedTensor]
    dropped_len: int

    # Selection metadata for debugging / experiment logging. These are aggregate values
    # across batch and heads; they do not affect decode attention.
    selection_mode: str = "recency"
    snap_score_weight: float = 0.0
    recency_weight: float = 1.0
    selected_mean_snap_score: float = 0.0
    selected_mean_recency: float = 0.0
    selected_mean_priority: float = 0.0

    def nbytes(self) -> int:
        total = 0
        if self.key_q is not None:
            total += self.key_q.nbytes()
        if self.value_q is not None:
            total += self.value_q.nbytes()
        return total

    def num_tokens(self) -> int:
        return int(self.dropped_len)


class SnapKVClusterKIVI:
    """
    SnapKV selector + KIVI-style storage for tokens that SnapKV would have discarded.

    Output of update_kv(...):
        retained_key/value: dense full-precision cache used by HuggingFace DynamicCache.
        dropped_cache: quantized cache for the old tokens not selected by SnapKV.

    Important: the dropped quantized cache is meant for decoding-time compensation. During prefill, compute
    attention with the original full K/V and only store the compressed/quantized cache for future decode.
    """
    def __init__(
        self,
        window_size: int = 64,
        max_capacity_prompt: int = 256 + 64,
        kernel_size: int = 5,
        pooling: str = "avgpool",
        kivi_bits: int = 2,
        kivi_group_size: int = 32,
        keep_dropped_quant: bool = True,
        kivi_max_capacity: int = -1,
        kivi_selection_mode: str = "score_recency",
        kivi_snap_score_weight: float = 0.7,
        kivi_recency_weight: float = 0.3,
        kivi_recency_power: float = 1.0,
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.keep_dropped_quant = keep_dropped_quant
        # Maximum number of SnapKV-dropped tokens kept in the KIVI quantized side cache.
        # -1 means unlimited. 0 means do not keep any dropped tokens.
        self.kivi_max_capacity = int(kivi_max_capacity)

        # KIVI side-cache selection policy:
        #   "recency"        : reproduce the old behaviour, keep the newest candidates.
        #   "snapkv"         : keep candidates with highest SnapKV pooled-attention score.
        #   "score_recency"  : weighted combination of normalized SnapKV score and recency.
        self.kivi_selection_mode = str(kivi_selection_mode).lower()
        if self.kivi_selection_mode not in {"recency", "snapkv", "score_recency"}:
            raise ValueError(
                "kivi_selection_mode must be one of {'recency', 'snapkv', 'score_recency'}, "
                f"got {kivi_selection_mode!r}"
            )
        self.kivi_snap_score_weight = float(kivi_snap_score_weight)
        self.kivi_recency_weight = float(kivi_recency_weight)
        self.kivi_recency_power = float(kivi_recency_power)
        if self.kivi_recency_power <= 0:
            raise ValueError("kivi_recency_power must be > 0")

        self.quantizer = KIVIUniformQuantizer(bits=kivi_bits, group_size=kivi_group_size)

    def reset(
        self,
        window_size: int = 64,
        max_capacity_prompt: int = 256 + 64,
        kernel_size: int = 5,
        pooling: str = "avgpool",
        kivi_bits: int = 2,
        kivi_group_size: int = 32,
        keep_dropped_quant: bool = True,
        kivi_max_capacity: int = -1,
        kivi_selection_mode: str = "score_recency",
        kivi_snap_score_weight: float = 0.7,
        kivi_recency_weight: float = 0.3,
        kivi_recency_power: float = 1.0,
    ):
        self.__init__(
            window_size,
            max_capacity_prompt,
            kernel_size,
            pooling,
            kivi_bits,
            kivi_group_size,
            keep_dropped_quant,
            kivi_max_capacity,
            kivi_selection_mode,
            kivi_snap_score_weight,
            kivi_recency_weight,
            kivi_recency_power,
        )

    @torch.no_grad()
    def _select_kivi_candidates(
        self,
        candidate_idx: torch.Tensor,
        candidate_scores: torch.Tensor,
        old_len: int,
        budget: int,
    ):
        """
        Select KIVI-side-cache token indices among SnapKV-dropped candidates.

        Args:
            candidate_idx:    [B, H, N] original token positions, sorted chronologically.
            candidate_scores: [B, H, N] SnapKV pooled-attention scores aligned with candidate_idx.
            old_len:          number of tokens before the observation window.
            budget:           number to keep; -1 means keep all.

        Returns:
            selected_idx in chronological order plus score/recency/priority tensors for logging.
        """
        n = candidate_idx.shape[-1]
        if budget < 0 or budget >= n:
            keep_n = n
        else:
            keep_n = max(0, int(budget))
        if keep_n == 0:
            empty = candidate_idx[..., :0]
            empty_f = candidate_scores[..., :0]
            return empty, empty_f, empty_f, empty_f

        # Normalize SnapKV score within each [batch, head] candidate set so it is on [0, 1]
        # before mixing with recency. This prevents the absolute attention scale from deciding
        # the coefficient balance across layers/heads.
        score_min = candidate_scores.amin(dim=-1, keepdim=True)
        score_max = candidate_scores.amax(dim=-1, keepdim=True)
        score_norm = (candidate_scores - score_min) / (score_max - score_min).clamp_min(1e-12)

        # Oldest candidate has recency 0; newest candidate has recency 1.
        if old_len <= 1:
            recency = torch.ones_like(score_norm)
        else:
            recency = candidate_idx.to(torch.float32) / float(old_len - 1)
        recency = recency.clamp_(0.0, 1.0).pow(self.kivi_recency_power)

        if self.kivi_selection_mode == "recency":
            priority = recency
        elif self.kivi_selection_mode == "snapkv":
            priority = score_norm
        else:
            w_score = max(0.0, self.kivi_snap_score_weight)
            w_recency = max(0.0, self.kivi_recency_weight)
            weight_sum = w_score + w_recency
            if weight_sum <= 0:
                raise ValueError(
                    "For kivi_selection_mode='score_recency', at least one of "
                    "kivi_snap_score_weight or kivi_recency_weight must be positive."
                )
            w_score /= weight_sum
            w_recency /= weight_sum
            priority = w_score * score_norm + w_recency * recency

        # Select by priority, then restore chronological order. The latter is not required for
        # RoPE-applied keys mathematically, but keeps K/V cache inspection and debugging sane.
        top_local = priority.topk(keep_n, dim=-1, largest=True, sorted=False).indices
        selected_idx = candidate_idx.gather(dim=-1, index=top_local)
        selected_raw_score = candidate_scores.gather(dim=-1, index=top_local)
        selected_recency = recency.gather(dim=-1, index=top_local)
        selected_priority = priority.gather(dim=-1, index=top_local)

        chrono_order = selected_idx.argsort(dim=-1)
        selected_idx = selected_idx.gather(dim=-1, index=chrono_order)
        selected_raw_score = selected_raw_score.gather(dim=-1, index=chrono_order)
        selected_recency = selected_recency.gather(dim=-1, index=chrono_order)
        selected_priority = selected_priority.gather(dim=-1, index=chrono_order)
        return selected_idx, selected_raw_score, selected_recency, selected_priority

    @torch.no_grad()
    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        # key/query/value are expected to be [B, num_attention_heads, T, D], because the supplied hijack
        # repeats K/V before calling this function, matching the user's current SnapKV implementation.
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        if q_len < self.max_capacity_prompt:
            return key_states, value_states, None

        attn_weights = torch.matmul(
            query_states[..., -self.window_size:, :], key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)

        # Causal mask for the observation window itself.
        mask = torch.full(
            (self.window_size, self.window_size),
            torch.finfo(attn_weights.dtype).min,
            device=attn_weights.device,
        )
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        attn_weights[:, :, -self.window_size:, -self.window_size:] += mask[None, None, :, :]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        old_len = q_len - self.window_size
        keep_len = self.max_capacity_prompt - self.window_size
        drop_len = old_len - keep_len
        if drop_len <= 0:
            return key_states, value_states, None

        attn_weights_sum = attn_weights[:, :, -self.window_size:, :old_len].sum(dim=-2)
        if self.pooling == "avgpool":
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        elif self.pooling == "maxpool":
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        else:
            raise ValueError("Pooling method not supported")

        keep_idx = attn_cache.topk(keep_len, dim=-1).indices  # [B,H,keep_len]
        keep_idx_exp = keep_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        old_k = key_states[:, :, :old_len, :]
        old_v = value_states[:, :, :old_len, :]
        k_past_keep = old_k.gather(dim=2, index=keep_idx_exp)
        v_past_keep = old_v.gather(dim=2, index=keep_idx_exp)

        dropped_cache = None
        if self.keep_dropped_quant:
            max_drop = int(self.kivi_max_capacity)
            if max_drop != 0:
                keep_mask = torch.zeros((bsz, num_heads, old_len), dtype=torch.bool, device=key_states.device)
                keep_mask.scatter_(dim=-1, index=keep_idx, value=True)

                # Candidate set = all old tokens not already retained in full precision by SnapKV.
                # Construct it in chronological order, then score/select it independently per [batch, head].
                all_idx = torch.arange(old_len, device=key_states.device).view(1, 1, old_len).expand(bsz, num_heads, old_len)
                drop_rank = all_idx.masked_fill(keep_mask, old_len)
                candidate_idx = torch.sort(drop_rank, dim=-1).values[..., :drop_len]
                candidate_scores = attn_cache.gather(dim=-1, index=candidate_idx)

                selected_idx, selected_scores, selected_recency, selected_priority = self._select_kivi_candidates(
                    candidate_idx=candidate_idx,
                    candidate_scores=candidate_scores,
                    old_len=old_len,
                    budget=max_drop,
                )
                drop_len_kept = selected_idx.shape[-1]

                if drop_len_kept > 0:
                    drop_idx_exp = selected_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    k_drop = old_k.gather(dim=2, index=drop_idx_exp).contiguous()
                    v_drop = old_v.gather(dim=2, index=drop_idx_exp).contiguous()
                    dropped_cache = SnapKVKIVIDroppedCache(
                        key_q=self.quantizer.quantize_key_per_channel(k_drop),
                        value_q=self.quantizer.quantize_value_per_token(v_drop),
                        dropped_len=drop_len_kept,
                        selection_mode=self.kivi_selection_mode,
                        snap_score_weight=self.kivi_snap_score_weight,
                        recency_weight=self.kivi_recency_weight,
                        selected_mean_snap_score=float(selected_scores.mean().detach().cpu()),
                        selected_mean_recency=float(selected_recency.mean().detach().cpu()),
                        selected_mean_priority=float(selected_priority.mean().detach().cpu()),
                    )

        k_cur = key_states[:, :, -self.window_size:, :]
        v_cur = value_states[:, :, -self.window_size:, :]
        key_retained = torch.cat([k_past_keep, k_cur], dim=2).contiguous()
        value_retained = torch.cat([v_past_keep, v_cur], dim=2).contiguous()
        return key_retained, value_retained, dropped_cache


def init_snapkv(self):
    if not hasattr(self.config, "window_size"):
        self.config.window_size = 32
    if not hasattr(self.config, "max_capacity_prompt"):
        self.config.max_capacity_prompt = 2048
    if not hasattr(self.config, "kernel_size"):
        self.config.kernel_size = 5
    if not hasattr(self.config, "pooling"):
        self.config.pooling = "avgpool"
    if not hasattr(self.config, "kivi_bits"):
        self.config.kivi_bits = 2
    if not hasattr(self.config, "kivi_group_size"):
        self.config.kivi_group_size = 32
    if not hasattr(self.config, "snapkv_quant_dropped"):
        self.config.snapkv_quant_dropped = True
    if not hasattr(self.config, "kivi_max_capacity"):
        self.config.kivi_max_capacity = -1
    if not hasattr(self.config, "kivi_selection_mode"):
        self.config.kivi_selection_mode = "score_recency"
    if not hasattr(self.config, "kivi_snap_score_weight"):
        self.config.kivi_snap_score_weight = 0.7
    if not hasattr(self.config, "kivi_recency_weight"):
        self.config.kivi_recency_weight = 0.3
    if not hasattr(self.config, "kivi_recency_power"):
        self.config.kivi_recency_power = 1.0
    if not hasattr(self.config, "snapkv_kivi_print_stats"):
        self.config.snapkv_kivi_print_stats = False

    # Recreate each call so changes to config during experiments take effect.
    self.kv_cluster = SnapKVClusterKIVI(
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        kernel_size=self.config.kernel_size,
        pooling=self.config.pooling,
        kivi_bits=self.config.kivi_bits,
        kivi_group_size=self.config.kivi_group_size,
        keep_dropped_quant=self.config.snapkv_quant_dropped,
        kivi_max_capacity=self.config.kivi_max_capacity,
        kivi_selection_mode=self.config.kivi_selection_mode,
        kivi_snap_score_weight=self.config.kivi_snap_score_weight,
        kivi_recency_weight=self.config.kivi_recency_weight,
        kivi_recency_power=self.config.kivi_recency_power,
    )
    if not hasattr(self, "snapkv_kivi_dropped_cache"):
        self.snapkv_kivi_dropped_cache = None



def _tensor_nbytes(t: Optional[torch.Tensor]) -> int:
    if t is None:
        return 0
    return int(t.numel() * t.element_size())


def format_bytes(num_bytes: int) -> str:
    num_bytes = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0 or unit == "TB":
            return f"{num_bytes:.2f}{unit}"
        num_bytes /= 1024.0


def collect_snapkv_kivi_cache_stats(model):
    """Collect per-layer and total KV cache stats for SnapKV retained cache and KIVI side cache.

    Call this after each model.generate(...) or at any point during generation:
        stats = collect_snapkv_kivi_cache_stats(model)

    The dense/SnapKV numbers are updated by the Mistral hijack forward after every cache update.
    """
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None and hasattr(model, "model") and hasattr(model.model, "model"):
        layers = getattr(model.model.model, "layers", None)
    if layers is None:
        raise ValueError("Could not find model.model.layers. Pass the HF causal LM model object.")

    per_layer = []
    total_snapkv_tokens = 0
    total_snapkv_bytes = 0
    total_kivi_tokens = 0
    total_kivi_bytes = 0

    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        dense_tokens = int(getattr(attn, "snapkv_dense_tokens", 0))
        dense_bytes = int(getattr(attn, "snapkv_dense_bytes", 0))
        dropped_cache = getattr(attn, "snapkv_kivi_dropped_cache", None)
        if dropped_cache is not None:
            kivi_tokens = int(dropped_cache.num_tokens())
            kivi_bytes = int(dropped_cache.nbytes())
            kivi_selection = {
                "mode": getattr(dropped_cache, "selection_mode", "recency"),
                "snap_score_weight": float(getattr(dropped_cache, "snap_score_weight", 0.0)),
                "recency_weight": float(getattr(dropped_cache, "recency_weight", 1.0)),
                "selected_mean_snap_score": float(getattr(dropped_cache, "selected_mean_snap_score", 0.0)),
                "selected_mean_recency": float(getattr(dropped_cache, "selected_mean_recency", 0.0)),
                "selected_mean_priority": float(getattr(dropped_cache, "selected_mean_priority", 0.0)),
            }
        else:
            kivi_tokens = 0
            kivi_bytes = 0
            kivi_selection = None
        logical_tokens = int(getattr(attn, "kv_seq_len", dense_tokens + kivi_tokens))
        row = {
            "layer": i,
            "logical_tokens": logical_tokens,
            "snapkv_tokens": dense_tokens,
            "snapkv_bytes": dense_bytes,
            "snapkv_human": format_bytes(dense_bytes),
            "kivi_tokens": kivi_tokens,
            "kivi_bytes": kivi_bytes,
            "kivi_human": format_bytes(kivi_bytes),
            "kivi_selection": kivi_selection,
            "total_bytes": dense_bytes + kivi_bytes,
            "total_human": format_bytes(dense_bytes + kivi_bytes),
        }
        per_layer.append(row)
        total_snapkv_tokens += dense_tokens
        total_snapkv_bytes += dense_bytes
        total_kivi_tokens += kivi_tokens
        total_kivi_bytes += kivi_bytes

    return {
        "layers": per_layer,
        "total_snapkv_tokens": total_snapkv_tokens,
        "total_snapkv_bytes": total_snapkv_bytes,
        "total_snapkv_human": format_bytes(total_snapkv_bytes),
        "total_kivi_tokens": total_kivi_tokens,
        "total_kivi_bytes": total_kivi_bytes,
        "total_kivi_human": format_bytes(total_kivi_bytes),
        "total_tokens": total_snapkv_tokens + total_kivi_tokens,
        "total_bytes": total_snapkv_bytes + total_kivi_bytes,
        "total_human": format_bytes(total_snapkv_bytes + total_kivi_bytes),
    }


def print_snapkv_kivi_cache_stats(model, every_layer: bool = False, prefix: str = "[SnapKV+KIVI]"):
    stats = collect_snapkv_kivi_cache_stats(model)
    print(
        f"{prefix} total: "
        f"snapkv={stats['total_snapkv_tokens']} tok/{stats['total_snapkv_human']}, "
        f"kivi={stats['total_kivi_tokens']} tok/{stats['total_kivi_human']}, "
        f"combined={stats['total_tokens']} tok/{stats['total_human']}"
    )
    if every_layer:
        for row in stats["layers"]:
            print(
                f"{prefix} layer {row['layer']:02d}: logical={row['logical_tokens']} | "
                f"snapkv={row['snapkv_tokens']} tok/{row['snapkv_human']} | "
                f"kivi={row['kivi_tokens']} tok/{row['kivi_human']} | "
                f"total={row['total_human']}"
            )
    return stats
