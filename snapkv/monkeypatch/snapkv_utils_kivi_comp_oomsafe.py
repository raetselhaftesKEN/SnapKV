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
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.keep_dropped_quant = keep_dropped_quant
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
    ):
        self.__init__(window_size, max_capacity_prompt, kernel_size, pooling, kivi_bits, kivi_group_size, keep_dropped_quant)

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
            keep_mask = torch.zeros((bsz, num_heads, old_len), dtype=torch.bool, device=key_states.device)
            keep_mask.scatter_(dim=-1, index=keep_idx, value=True)
            # Exactly drop_len False positions per [B,H]. Order is not semantically important for decoding
            # because keys are already RoPE-applied and values are position-independent.
            drop_idx = torch.argsort(keep_mask.to(torch.int8), dim=-1)[..., :drop_len]
            drop_idx_exp = drop_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_drop = old_k.gather(dim=2, index=drop_idx_exp).contiguous()
            v_drop = old_v.gather(dim=2, index=drop_idx_exp).contiguous()
            dropped_cache = SnapKVKIVIDroppedCache(
                key_q=self.quantizer.quantize_key_per_channel(k_drop),
                value_q=self.quantizer.quantize_value_per_token(v_drop),
                dropped_len=drop_len,
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

    # Recreate each call so changes to config during experiments take effect.
    self.kv_cluster = SnapKVClusterKIVI(
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        kernel_size=self.config.kernel_size,
        pooling=self.config.pooling,
        kivi_bits=self.config.kivi_bits,
        kivi_group_size=self.config.kivi_group_size,
        keep_dropped_quant=self.config.snapkv_quant_dropped,
    )
    if not hasattr(self, "snapkv_kivi_dropped_cache"):
        self.snapkv_kivi_dropped_cache = None
