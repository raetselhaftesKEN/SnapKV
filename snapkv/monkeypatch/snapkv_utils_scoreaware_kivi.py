import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

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
        if bits not in (1, 2, 4, 8):
            raise ValueError(f"bits must be 1, 2, 4, or 8, got {bits}")
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
        if self.bits == 2:
            if q.shape[-1] % 4 != 0:
                q = F.pad(q, (0, 4 - q.shape[-1] % 4), value=0)
            q = q.view(*q.shape[:-1], q.shape[-1] // 4, 4)
            return (q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)).contiguous()
        # bits == 1, one byte stores 8 binary values.
        if q.shape[-1] % 8 != 0:
            q = F.pad(q, (0, 8 - q.shape[-1] % 8), value=0)
        q = q.view(*q.shape[:-1], q.shape[-1] // 8, 8)
        return (
            q[..., 0]
            | (q[..., 1] << 1)
            | (q[..., 2] << 2)
            | (q[..., 3] << 3)
            | (q[..., 4] << 4)
            | (q[..., 5] << 5)
            | (q[..., 6] << 6)
            | (q[..., 7] << 7)
        ).contiguous()

    def _unpack_uint8(self, packed: torch.Tensor, unpacked_last_dim: int) -> torch.Tensor:
        packed = packed.to(torch.uint8).contiguous()
        if self.bits == 8:
            q = packed
        elif self.bits == 4:
            q0 = packed & 0x0F
            q1 = (packed >> 4) & 0x0F
            q = torch.stack((q0, q1), dim=-1).flatten(-2)
        elif self.bits == 2:
            q0 = packed & 0x03
            q1 = (packed >> 2) & 0x03
            q2 = (packed >> 4) & 0x03
            q3 = (packed >> 6) & 0x03
            q = torch.stack((q0, q1, q2, q3), dim=-1).flatten(-2)
        else:
            q0 = packed & 0x01
            q1 = (packed >> 1) & 0x01
            q2 = (packed >> 2) & 0x01
            q3 = (packed >> 3) & 0x01
            q4 = (packed >> 4) & 0x01
            q5 = (packed >> 5) & 0x01
            q6 = (packed >> 6) & 0x01
            q7 = (packed >> 7) & 0x01
            q = torch.stack((q0, q1, q2, q3, q4, q5, q6, q7), dim=-1).flatten(-2)
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
class SnapKVKIVITier:
    """One score-aware quantization tier for SnapKV-dropped tokens."""
    name: str
    key_q: Optional[KIVIQuantizedTensor]
    value_q: Optional[KIVIQuantizedTensor]
    dropped_len: int
    k_bits: int
    v_bits: int
    k_group_size: int
    v_group_size: int
    score_min: float = 0.0
    score_max: float = 0.0

    def nbytes(self) -> int:
        total = 0
        if self.key_q is not None:
            total += self.key_q.nbytes()
        if self.value_q is not None:
            total += self.value_q.nbytes()
        return total

    def num_tokens(self) -> int:
        return int(self.dropped_len)


@dataclass
class SnapKVKIVIDroppedCache:
    tiers: List[SnapKVKIVITier]

    @property
    def dropped_len(self) -> int:
        return self.num_tokens()

    def nbytes(self) -> int:
        return sum(t.nbytes() for t in self.tiers)

    def num_tokens(self) -> int:
        return sum(t.num_tokens() for t in self.tiers)

    def tier_summary(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "tokens": t.num_tokens(),
                "bytes": t.nbytes(),
                "human": format_bytes(t.nbytes()) if "format_bytes" in globals() else str(t.nbytes()),
                "k_bits": t.k_bits,
                "v_bits": t.v_bits,
                "k_group_size": t.k_group_size,
                "v_group_size": t.v_group_size,
                "score_min": t.score_min,
                "score_max": t.score_max,
            }
            for t in self.tiers
        ]


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
        kivi_tiers: Optional[List[Dict[str, Any]]] = None,
        kivi_total_token_budget: int = -1,
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.keep_dropped_quant = keep_dropped_quant
        # Maximum number of SnapKV-dropped tokens kept in the KIVI quantized side cache.
        # -1 means unlimited. 0 means do not keep any dropped tokens. If the number of dropped
        # tokens exceeds this capacity, the oldest dropped tokens are discarded and only the
        # newest kivi_max_capacity dropped tokens are quantized/stored.
        self.kivi_max_capacity = int(kivi_max_capacity)
        self.kivi_total_token_budget = int(kivi_total_token_budget if kivi_total_token_budget is not None else kivi_max_capacity)
        self.kivi_tiers = self._normalize_tiers(kivi_tiers, default_bits=kivi_bits, default_group_size=kivi_group_size)
        # Kept for backward compatibility with old single-level callers. New score-aware code creates
        # per-tier key/value quantizers because K and V may use different bit/group settings.
        self.quantizer = KIVIUniformQuantizer(bits=kivi_bits, group_size=kivi_group_size)

    @staticmethod
    def _normalize_tiers(kivi_tiers: Optional[List[Dict[str, Any]]], default_bits: int = 2, default_group_size: int = 32):
        """Normalize user-provided tier specs.

        Default policy:
          QLevel1: top 50% dropped tokens, K/V 2bit group 32
          QLevel2: bottom 50% dropped tokens, K/V 1bit group 64
        """
        if kivi_tiers is None:
            kivi_tiers = [
                {"name": "QLevel1", "ratio": 0.5, "k_bits": 2, "v_bits": 2, "k_group_size": 32, "v_group_size": 32},
                {"name": "QLevel2", "ratio": 0.5, "k_bits": 1, "v_bits": 1, "k_group_size": 64, "v_group_size": 64},
            ]
        out = []
        for i, spec in enumerate(kivi_tiers):
            spec = dict(spec)
            ratio = float(spec.get("ratio", 0.0))
            if ratio < 0:
                raise ValueError(f"tier ratio must be non-negative, got {ratio} for tier {spec}")
            k_bits = int(spec.get("k_bits", spec.get("bits", default_bits)))
            v_bits = int(spec.get("v_bits", spec.get("bits", default_bits)))
            k_group = int(spec.get("k_group_size", spec.get("group_size", default_group_size)))
            v_group = int(spec.get("v_group_size", spec.get("group_size", default_group_size)))
            out.append({
                "name": str(spec.get("name", f"QLevel{i+1}")),
                "ratio": ratio,
                "k_bits": k_bits,
                "v_bits": v_bits,
                "k_group_size": k_group,
                "v_group_size": v_group,
            })
        ratio_sum = sum(x["ratio"] for x in out)
        if ratio_sum <= 0:
            raise ValueError("sum of tier ratios must be positive")
        for x in out:
            x["ratio"] = x["ratio"] / ratio_sum
        return out

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
        kivi_tiers: Optional[List[Dict[str, Any]]] = None,
        kivi_total_token_budget: int = -1,
    ):
        self.__init__(window_size, max_capacity_prompt, kernel_size, pooling, kivi_bits, kivi_group_size, keep_dropped_quant, kivi_max_capacity, kivi_tiers, kivi_total_token_budget)

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
            # Score-aware bit allocation. Dropped tokens are sorted by SnapKV score descending;
            # higher-score tokens receive earlier/higher-fidelity tiers. If a token budget is set,
            # budget is consumed from high to low tier, so the lowest-score QLevel2 tokens are dropped first.
            budget = int(self.kivi_total_token_budget)
            if budget < 0:
                # Backward-compatible fallback: old kivi_max_capacity also serves as token budget if set.
                budget = int(self.kivi_max_capacity)
            if budget != 0:
                keep_mask = torch.zeros((bsz, num_heads, old_len), dtype=torch.bool, device=key_states.device)
                keep_mask.scatter_(dim=-1, index=keep_idx, value=True)
                drop_scores = attn_cache.masked_fill(keep_mask, torch.finfo(attn_cache.dtype).min)
                sorted_drop_idx = torch.argsort(drop_scores, dim=-1, descending=True)[..., :drop_len]
                sorted_drop_scores = drop_scores.gather(dim=-1, index=sorted_drop_idx)

                # Original tier counts are based on all dropped tokens, before applying total budget.
                tier_counts = []
                used = 0
                for ti, spec in enumerate(self.kivi_tiers):
                    if ti == len(self.kivi_tiers) - 1:
                        cnt = drop_len - used
                    else:
                        cnt = int(round(drop_len * float(spec["ratio"])))
                        cnt = max(0, min(cnt, drop_len - used))
                    tier_counts.append(cnt)
                    used += cnt

                remaining_budget = drop_len if budget < 0 else min(budget, drop_len)
                start_pos = 0
                tiers = []
                for spec, cnt in zip(self.kivi_tiers, tier_counts):
                    if cnt <= 0:
                        continue
                    keep_cnt = min(cnt, remaining_budget)
                    remaining_budget -= keep_cnt
                    if keep_cnt <= 0:
                        start_pos += cnt
                        continue

                    tier_idx = sorted_drop_idx[..., start_pos : start_pos + keep_cnt]
                    tier_score = sorted_drop_scores[..., start_pos : start_pos + keep_cnt]
                    idx_exp = tier_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    k_drop = old_k.gather(dim=2, index=idx_exp).contiguous()
                    v_drop = old_v.gather(dim=2, index=idx_exp).contiguous()

                    k_bits = int(spec["k_bits"])
                    v_bits = int(spec["v_bits"])
                    k_group = int(spec["k_group_size"])
                    v_group = int(spec["v_group_size"])
                    k_quantizer = KIVIUniformQuantizer(bits=k_bits, group_size=k_group)
                    v_quantizer = KIVIUniformQuantizer(bits=v_bits, group_size=v_group)
                    tiers.append(
                        SnapKVKIVITier(
                            name=str(spec["name"]),
                            key_q=k_quantizer.quantize_key_per_channel(k_drop),
                            value_q=v_quantizer.quantize_value_per_token(v_drop),
                            dropped_len=keep_cnt,
                            k_bits=k_bits,
                            v_bits=v_bits,
                            k_group_size=k_group,
                            v_group_size=v_group,
                            score_min=float(tier_score.min().detach().cpu()) if tier_score.numel() else 0.0,
                            score_max=float(tier_score.max().detach().cpu()) if tier_score.numel() else 0.0,
                        )
                    )
                    start_pos += cnt
                    if remaining_budget <= 0:
                        break

                if len(tiers) > 0:
                    dropped_cache = SnapKVKIVIDroppedCache(tiers=tiers)

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
    if not hasattr(self.config, "kivi_total_token_budget"):
        self.config.kivi_total_token_budget = self.config.kivi_max_capacity
    if not hasattr(self.config, "kivi_tiers"):
        self.config.kivi_tiers = None
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
        kivi_tiers=self.config.kivi_tiers,
        kivi_total_token_budget=self.config.kivi_total_token_budget,
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
            kivi_tiers = dropped_cache.tier_summary() if hasattr(dropped_cache, "tier_summary") else []
        else:
            kivi_tokens = 0
            kivi_bytes = 0
            kivi_tiers = []
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
            "kivi_tiers": kivi_tiers,
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
            for tier in row.get("kivi_tiers", []):
                print(
                    f"{prefix}   - {tier['name']}: {tier['tokens']} tok/{tier['human']} | "
                    f"K{tier['k_bits']}g{tier['k_group_size']} V{tier['v_bits']}g{tier['v_group_size']} | "
                    f"score=[{tier['score_min']:.4g},{tier['score_max']:.4g}]"
                )
    return stats
