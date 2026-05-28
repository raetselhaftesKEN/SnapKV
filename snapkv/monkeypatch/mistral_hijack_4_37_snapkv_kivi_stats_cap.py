import inspect
import math
import torch
import torch.nn.functional as F
from typing import Optional
import warnings
from transformers.cache_utils import Cache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, repeat_kv
from transformers.utils import logging, is_flash_attn_2_available

# Put snapkv_utils_kivi_comp.py under snapkv/monkeypatch/ and import from there.
from snapkv.monkeypatch.snapkv_utils import init_snapkv, _tensor_nbytes, format_bytes

logger = logging.get_logger(__name__)


def _update_snapkv_kivi_layer_stats(self, dense_key_states=None, dense_value_states=None):
    """Update lightweight per-layer cache statistics.

    SnapKV dense cache lives in HuggingFace DynamicCache, so the hijack must record its
    current shape/bytes whenever past_key_value.update(...) returns it. KIVI dropped cache
    lives on self.snapkv_kivi_dropped_cache and exposes nbytes().
    """
    if dense_key_states is not None and dense_value_states is not None:
        self.snapkv_dense_tokens = int(dense_key_states.shape[-2])
        self.snapkv_dense_bytes = _tensor_nbytes(dense_key_states) + _tensor_nbytes(dense_value_states)
    elif not hasattr(self, "snapkv_dense_tokens"):
        self.snapkv_dense_tokens = 0
        self.snapkv_dense_bytes = 0

    dropped_cache = getattr(self, "snapkv_kivi_dropped_cache", None)
    if dropped_cache is not None:
        self.snapkv_kivi_tokens = int(dropped_cache.num_tokens())
        self.snapkv_kivi_bytes = int(dropped_cache.nbytes())
    else:
        self.snapkv_kivi_tokens = 0
        self.snapkv_kivi_bytes = 0

    if getattr(self.config, "snapkv_kivi_print_stats", False):
        # Per-layer printing is verbose but useful for debugging. For per-sample summaries,
        # prefer print_snapkv_kivi_cache_stats(model) after model.generate(...).
        print(
            f"[SnapKV+KIVI][layer {getattr(self, 'layer_idx', -1)}] "
            f"logical={int(getattr(self, 'kv_seq_len', 0))} | "
            f"snapkv={self.snapkv_dense_tokens} tok/{format_bytes(self.snapkv_dense_bytes)} | "
            f"kivi={self.snapkv_kivi_tokens} tok/{format_bytes(self.snapkv_kivi_bytes)} | "
            f"total={format_bytes(self.snapkv_dense_bytes + self.snapkv_kivi_bytes)}"
        )

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
else:
    _flash_supports_window_size = False


def _manual_attn_with_optional_quant_dropped(self, query_states, dense_key_states, dense_value_states, dropped_cache):
    """
    OOM-safe exact decode attention over:
        quantized dropped KV + dense retained KV.

    This avoids materializing:
        all_k = cat([dequant(k_drop), dense_k])
        all_v = cat([dequant(v_drop), dense_v])
    which can easily OOM on 24GB GPUs. Instead it dequantizes dropped KV in token chunks
    and computes a numerically stable two-pass softmax.

    Expected decode shape: query_states [B,H,1,D]. Prefill is handled by FlashAttention before
    the cache is compressed, so q_len should normally be 1 here.
    """
    bsz, num_heads, q_len, head_dim = query_states.shape
    dtype = query_states.dtype

    if dropped_cache is None or dropped_cache.dropped_len <= 0:
        attn_weights = torch.matmul(query_states, dense_key_states.transpose(2, 3)) / math.sqrt(head_dim)
        if q_len > 1:
            kv_len = dense_key_states.shape[-2]
            causal = torch.full((q_len, kv_len), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            causal = torch.triu(causal, diagonal=1 + kv_len - q_len)
            attn_weights = attn_weights + causal[None, None, :, :]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
        return torch.matmul(attn_weights, dense_value_states), attn_weights

    if q_len != 1:
        raise RuntimeError(
            "SnapKV+KIVI OOM-safe path currently expects q_len=1 during decoding. "
            "Prefill should use the full FlashAttention path before cache compression."
        )

    qtz = self.kv_cluster.quantizer
    scale = 1.0 / math.sqrt(head_dim)
    dropped_len = int(dropped_cache.dropped_len)
    # Smaller chunks lower peak memory; larger chunks are faster. 256/512 is usually safe on 24GB GPUs.
    chunk_size = int(getattr(self.config, "snapkv_kivi_chunk_size", 512))
    chunk_size = max(1, chunk_size)

    # Pass 1: compute global max logits across quantized dropped chunks and dense retained KV.
    max_logits = None  # [B,H,1,1], fp32
    for st in range(0, dropped_len, chunk_size):
        ed = min(st + chunk_size, dropped_len)
        k_chunk = qtz.dequantize_key_per_channel_range(dropped_cache.key_q, st, ed, dtype=dtype)
        logits = torch.matmul(query_states, k_chunk.transpose(2, 3)) * scale
        m = logits.float().amax(dim=-1, keepdim=True)
        max_logits = m if max_logits is None else torch.maximum(max_logits, m)
        del k_chunk, logits, m

    dense_logits = torch.matmul(query_states, dense_key_states.transpose(2, 3)) * scale
    dense_max = dense_logits.float().amax(dim=-1, keepdim=True)
    max_logits = dense_max if max_logits is None else torch.maximum(max_logits, dense_max)
    del dense_max

    # Pass 2: accumulate exp(logits-max) * V and denominator. This is exactly equivalent to
    # softmax over concatenated [dropped, dense] KV, but without materializing the concatenation.
    denom = torch.zeros((bsz, num_heads, q_len, 1), device=query_states.device, dtype=torch.float32)
    out = torch.zeros((bsz, num_heads, q_len, head_dim), device=query_states.device, dtype=torch.float32)

    for st in range(0, dropped_len, chunk_size):
        ed = min(st + chunk_size, dropped_len)
        k_chunk = qtz.dequantize_key_per_channel_range(dropped_cache.key_q, st, ed, dtype=dtype)
        v_chunk = qtz.dequantize_value_per_token_range(dropped_cache.value_q, st, ed, dtype=dtype)
        logits = torch.matmul(query_states, k_chunk.transpose(2, 3)) * scale
        weights = torch.exp(logits.float() - max_logits)
        denom += weights.sum(dim=-1, keepdim=True)
        out += torch.matmul(weights.to(v_chunk.dtype), v_chunk).float()
        del k_chunk, v_chunk, logits, weights

    dense_weights = torch.exp(dense_logits.float() - max_logits)
    denom += dense_weights.sum(dim=-1, keepdim=True)
    out += torch.matmul(dense_weights.to(dense_value_states.dtype), dense_value_states).float()
    del dense_logits, dense_weights

    attn_output = (out / denom.clamp_min(1e-20)).to(dtype)
    # Returning full attention weights would itself allocate [B,H,1,total_len]. Keep None for memory safety.
    return attn_output, None


def mistral_flash_attn2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    # [SnapKV + KIVI] register kv_cluster
    init_snapkv(self)
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please use `attention_mask` instead."
        )
        attention_mask = kwargs.pop("padding_mask")

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError("layer_idx is required for autoregressive cache update.")
        if hasattr(self, "kv_seq_len"):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
    cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    use_sliding_windows = (
        _flash_supports_window_size
        and getattr(self.config, "sliding_window", None) is not None
        and kv_seq_len > self.config.sliding_window
    )
    if not _flash_supports_window_size:
        logger.warning_once(
            "The current flash-attn version does not support sliding-window attention."
        )

    # Match the user's current SnapKV implementation: repeat before selection/quantization.
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    dropout_rate = 0.0 if not self.training else self.attention_dropout
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        target_dtype = getattr(self.config, "_pre_quantization_dtype", self.q_proj.weight.dtype)
        logger.warning_once(f"Casting hidden states back to {target_dtype}.")
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_weights = None

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}

        # Prefill / first cache write: q_len is the prompt length. Compute prompt attention with the original
        # full K/V, then store retained full tokens + quantized dropped tokens for future decoding.
        if key_states.shape[-2] >= kv_seq_len:
            self.kv_seq_len = kv_seq_len

            full_q = query_states
            full_k = key_states
            full_v = value_states

            key_retained, value_retained, dropped_cache = self.kv_cluster.update_kv(
                full_k, full_q, full_v, attention_mask, self.num_key_value_groups
            )
            self.snapkv_kivi_dropped_cache = dropped_cache
            past_key_value.update(key_retained, value_retained, self.layer_idx, cache_kwargs)
            _update_snapkv_kivi_layer_stats(self, key_retained, value_retained)

            # Full prefill attention. This avoids computing the prompt hidden states with a compressed key/value set.
            q_flash = full_q.transpose(1, 2)
            k_flash = full_k.transpose(1, 2)
            v_flash = full_v.transpose(1, 2)
            attn_output = self._flash_attention_forward(
                q_flash,
                k_flash,
                v_flash,
                attention_mask,
                q_len,
                dropout=dropout_rate,
                use_sliding_windows=use_sliding_windows,
            )
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, None if not output_attentions else attn_weights, past_key_value

        # Decode: append the current token to the dense retained cache, then compensate with quantized dropped cache.
        self.kv_seq_len += q_len
        dense_key_states, dense_value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        _update_snapkv_kivi_layer_stats(self, dense_key_states, dense_value_states)
        dropped_cache = getattr(self, "snapkv_kivi_dropped_cache", None)

        if dropped_cache is not None and getattr(self.config, "snapkv_quant_dropped", True):
            attn_output_heads, attn_weights = _manual_attn_with_optional_quant_dropped(
                self, query_states, dense_key_states, dense_value_states, dropped_cache
            )
            attn_output = attn_output_heads.transpose(1, 2).reshape(bsz, q_len, self.hidden_size).contiguous()
            attn_output = self.o_proj(attn_output)
            if not output_attentions:
                attn_weights = None
            return attn_output, attn_weights, past_key_value

        # No dropped quantized cache, so use the original flash-attn path over dense cache.
        key_states, value_states = dense_key_states, dense_value_states

    # No cache, or no quantized dropped cache: original FlashAttention path.
    query_states_flash = query_states.transpose(1, 2)
    key_states_flash = key_states.transpose(1, 2)
    value_states_flash = value_states.transpose(1, 2)
    attn_output = self._flash_attention_forward(
        query_states_flash,
        key_states_flash,
        value_states_flash,
        attention_mask,
        q_len,
        dropout=dropout_rate,
        use_sliding_windows=use_sliding_windows,
    )
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)
    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value


def prepare_inputs_for_generation_mistral(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values is None:
        for layer in self.model.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.kv_seq_len = 0
                layer.self_attn.snapkv_kivi_dropped_cache = None
                layer.self_attn.snapkv_dense_tokens = 0
                layer.self_attn.snapkv_dense_bytes = 0
                layer.self_attn.snapkv_kivi_tokens = 0
                layer.self_attn.snapkv_kivi_bytes = 0

    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
            # In SnapKV+quantized-dropped mode, DynamicCache contains only retained dense tokens;
            # self_attn.kv_seq_len tracks the original logical sequence length.
            if hasattr(self.model.layers[0].self_attn, "kv_seq_len") and self.model.layers[0].self_attn.kv_seq_len != 0:
                cache_length = past_length = self.model.layers[0].self_attn.kv_seq_len
        else:
            cache_length = past_length = self.model.layers[0].self_attn.kv_seq_len
            max_cache_length = None

        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]

        if max_cache_length is not None and attention_mask is not None and cache_length + input_ids.shape[1] > max_cache_length:
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs
