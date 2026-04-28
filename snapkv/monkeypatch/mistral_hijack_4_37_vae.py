import inspect
import json
import os
import math
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, repeat_kv
from transformers.utils import logging, is_flash_attn_2_available

from snapkv.monkeypatch.snapkv_utils import init_snapkv

logger = logging.get_logger(__name__)

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

def _gather_tokens(x: torch.Tensor, token_idx: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D], token_idx: [B, K]
    return x.gather(2, token_idx[:, None, :, None].expand(-1, x.size(1), -1, x.size(3)))


def _apply_rope_to_k_only(key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    dummy_q = torch.zeros_like(key_states)
    _, k_out = apply_rotary_pos_emb(dummy_q, key_states, cos, sin, position_ids)
    return k_out


def _compress_dropped_to_latent(attn_module, dropped_k_pre: torch.Tensor, dropped_v_pre: torch.Tensor):
    """
    dropped_k_pre/dropped_v_pre: [B, num_kv_heads, Tdrop, Hd]
    返回 latent mu: [B, Tdrop, latent_dim]
    """
    bsz, num_kv_heads, seq_len, head_dim = dropped_k_pre.shape
    k_flat = dropped_k_pre.transpose(1, 2).contiguous().reshape(bsz, seq_len, num_kv_heads * head_dim)
    v_flat = dropped_v_pre.transpose(1, 2).contiguous().reshape(bsz, seq_len, num_kv_heads * head_dim)
    kv = torch.cat([k_flat, v_flat], dim=-1)

    with torch.no_grad():
        mu, logvar = attn_module.kv_vae.encode(kv)

    return mu


def _restore_dropped_from_latent(attn_module, latent: torch.Tensor, num_kv_heads: int, head_dim: int):
    """
    latent: [B, Tdrop, latent_dim]
    返回 pre-RoPE unrepeated KV:
      recon_k_pre / recon_v_pre: [B, num_kv_heads, Tdrop, Hd]
    """
    with torch.no_grad():
        recon_kv = attn_module.kv_vae.decode(latent)

    recon_k_flat, recon_v_flat = torch.split(recon_kv, num_kv_heads * head_dim, dim=-1)
    bsz, seq_len, _ = recon_k_flat.shape

    recon_k = recon_k_flat.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2).contiguous()
    recon_v = recon_v_flat.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2).contiguous()
    return recon_k, recon_v

# 新增 hybrid cache 构建函数
def _maybe_build_hybrid_cache(
    attn_module,
    key_states_pre: torch.Tensor,
    value_states_pre: torch.Tensor,
    key_states_rep_postrope: torch.Tensor,
    query_states_rep_postrope: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
):
    """
    prefix 阶段：
      - keep 的 token 以全精度 KV 存入 normal cache
      - dropped 的 token 只存 latent(mu) + positions
    """
    init_kv_vae_for_mistral_attn(attn_module)
    if not getattr(attn_module, 'use_kv_vae_cache', False):
        return None, None, False

    keep_idx, drop_idx = attn_module.kv_cluster.select_important_tokens(
        key_states_rep_postrope, query_states_rep_postrope
    )
    if keep_idx is None:
        return None, None, False

    bsz, num_kv_heads, q_len, head_dim = key_states_pre.shape
    window = attn_module.kv_cluster.window_size
    old_len = q_len - window
    device = key_states_pre.device

    # kept old tokens + current window
    keep_old_k_pre = _gather_tokens(key_states_pre[:, :, :old_len, :], keep_idx)
    keep_old_v_pre = _gather_tokens(value_states_pre[:, :, :old_len, :], keep_idx)
    cur_k_pre = key_states_pre[:, :, -window:, :]
    cur_v_pre = value_states_pre[:, :, -window:, :]

    kept_k_pre = torch.cat([keep_old_k_pre, cur_k_pre], dim=2)
    kept_v_pre = torch.cat([keep_old_v_pre, cur_v_pre], dim=2)

    keep_pos = torch.cat([
        keep_idx,
        torch.arange(old_len, q_len, device=device).unsqueeze(0).expand(bsz, -1)
    ], dim=1)

    kept_k_post = _apply_rope_to_k_only(kept_k_pre, cos, sin, keep_pos)
    kept_v_post = kept_v_pre

    kept_k_rep = repeat_kv(kept_k_post, attn_module.num_key_value_groups)
    kept_v_rep = repeat_kv(kept_v_post, attn_module.num_key_value_groups)

    # dropped old tokens -> latent only
    if drop_idx.size(1) > 0:
        drop_k_pre = _gather_tokens(key_states_pre[:, :, :old_len, :], drop_idx)
        drop_v_pre = _gather_tokens(value_states_pre[:, :, :old_len, :], drop_idx)

        latent = _compress_dropped_to_latent(attn_module, drop_k_pre, drop_v_pre)

        attn_module.kv_cluster.store_vae_compressed(
            latent=latent,
            positions=drop_idx,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            group_size=attn_module.num_key_value_groups,
        )
    else:
        attn_module.kv_cluster.clear_vae_cache()

    return kept_k_rep, kept_v_rep, True

# 新增 decode 恢复函数
def _maybe_restore_dropped_cache(attn_module, cos, sin):
    init_kv_vae_for_mistral_attn(attn_module)
    if not getattr(attn_module, 'use_kv_vae_cache', False):
        return None, None
    if not attn_module.kv_cluster.has_vae_cache():
        return None, None

    latent = attn_module.kv_cluster.vae_dropped_latent.to(
        device=attn_module.q_proj.weight.device,
        dtype=attn_module.q_proj.weight.dtype
    )
    drop_pos = attn_module.kv_cluster.vae_dropped_positions.to(device=attn_module.q_proj.weight.device)
    num_kv_heads = attn_module.kv_cluster.vae_num_kv_heads
    head_dim = attn_module.kv_cluster.vae_head_dim

    recon_k_pre, recon_v_pre = _restore_dropped_from_latent(attn_module, latent, num_kv_heads, head_dim)

    recon_k_post = _apply_rope_to_k_only(recon_k_pre, cos, sin, drop_pos)
    recon_v_post = recon_v_pre

    recon_k_rep = repeat_kv(recon_k_post, attn_module.num_key_value_groups)
    recon_v_rep = repeat_kv(recon_v_post, attn_module.num_key_value_groups)

    return recon_k_rep, recon_v_rep


# ============================================================
# Predictor-friendly KV-VAE inference helpers
# ============================================================
class InferenceTokenVAE(nn.Module):
    """
    Inference-only mirror of train_mistral_kv_latent_predictor_friendly.py
    We only need deterministic encode(mu) + decode(mu) by default.
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, logvar_min: float = -4.0, logvar_max: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.enc_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_ln = nn.LayerNorm(hidden_dim)

        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)

        self.dec_in = nn.Linear(latent_dim, hidden_dim)
        self.dec_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dec_ln = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def encode_hidden(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.in_proj(x))
        h_res = h
        h = F.silu(self.enc_fc1(h))
        h = self.enc_fc2(h)
        h = self.enc_ln(h + h_res)
        h = F.silu(h)
        return h

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encode_hidden(x)
        mu = self.mu_proj(h)
        logvar_raw = self.logvar_proj(h)
        logvar = torch.clamp(logvar_raw, min=self.logvar_min, max=self.logvar_max)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.dec_in(z))
        h_res = h
        h = F.silu(self.dec_fc1(h))
        h = self.dec_fc2(h)
        h = self.dec_ln(h + h_res)
        h = F.silu(h)
        return self.out_proj(h)

    def forward(self, x: torch.Tensor, deterministic: bool = True):
        mu, logvar = self.encode(x)
        if deterministic:
            z = mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        recon = self.decode(z)
        return recon, mu, logvar, z


class SharedInferenceVAEPool(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.group_size = int(getattr(config, 'vae_group_size', 4))
        self.num_layers = int(config.num_hidden_layers)
        self.num_groups = math.ceil(self.num_layers / self.group_size)

        kv_dim = config.num_key_value_heads * (config.hidden_size // config.num_attention_heads)
        input_dim = 2 * kv_dim

        self.group_vaes = nn.ModuleList([
            InferenceTokenVAE(
                input_dim=input_dim,
                latent_dim=int(getattr(config, 'kv_latent_size', 32)),
                hidden_dim=int(getattr(config, 'vae_hidden_size', 256)),
                logvar_min=float(getattr(config, 'logvar_min', -4.0)),
                logvar_max=float(getattr(config, 'logvar_max', 1.0)),
            )
            for _ in range(self.num_groups)
        ])

    def get_vae(self, layer_idx: int) -> nn.Module:
        return self.group_vaes[layer_idx // self.group_size]


def _strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return {}
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _load_vae_group_state(vae_module: nn.Module, full_state: Dict[str, torch.Tensor], layer_idx: int, group_idx: int):
    candidates = [
        f"model.shared_vae_pool.group_vaes.{group_idx}.",
        f"shared_vae_pool.group_vaes.{group_idx}.",
        f"model.layers.{layer_idx}.self_attn.kv_vae.",
        f"layers.{layer_idx}.self_attn.kv_vae.",
    ]
    loaded = False
    for prefix in candidates:
        sub = _strip_prefix_if_present(full_state, prefix)
        if sub:
            missing, unexpected = vae_module.load_state_dict(sub, strict=False)
            logger.info(f"Loaded VAE weights for layer={layer_idx}, group={group_idx} from prefix='{prefix}', missing={len(missing)}, unexpected={len(unexpected)}")
            loaded = True
            break
    if not loaded:
        raise KeyError(
            f"Could not find VAE weights for layer {layer_idx} / group {group_idx}. "
            f"Tried prefixes: {candidates[:]}"
        )


def init_kv_vae_for_mistral_attn(self):
    """
    Lazy init on the attention module itself.

    Required config fields:
      config.vae_ckpt_path: path to trainable_vae_only.bin or checkpoint dir containing it

    Optional config fields:
      config.use_kv_vae: bool, default False
      config.kv_vae_deterministic: bool, default True
      config.share_vae_across_layers: bool, default True
      config.vae_group_size: int, default 4
      config.kv_latent_size, config.vae_hidden_size, config.logvar_min, config.logvar_max
      config.kv_vae_apply_on_decode_only: bool, default False
    """
    use_kv_vae = bool(getattr(self.config, 'use_kv_vae', False))
    if not use_kv_vae:
        self.use_kv_vae = False
        return

    if getattr(self, 'kv_vae_inited', False):
        return

    ckpt_path = getattr(self.config, 'vae_ckpt_path', None)
    if ckpt_path is None:
        raise ValueError('config.use_kv_vae=True but config.vae_ckpt_path is not set.')

    ckpt_path = str(ckpt_path)
    if os.path.isdir(ckpt_path):
        candidate = os.path.join(ckpt_path, 'trainable_vae_only.bin')
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Could not find trainable_vae_only.bin under directory: {ckpt_path}")
        ckpt_file = candidate
    else:
        ckpt_file = ckpt_path

    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"VAE checkpoint not found: {ckpt_file}")

    # optional extra config json next to checkpoint
    extra_cfg_path = os.path.join(os.path.dirname(ckpt_file), 'extra_vae_config.json')
    if os.path.exists(extra_cfg_path):
        with open(extra_cfg_path, 'r', encoding='utf-8') as f:
            extra_cfg = json.load(f)
        for k, v in extra_cfg.items():
            if not hasattr(self.config, k):
                setattr(self.config, k, v)

    state = torch.load(ckpt_file, map_location='cpu')
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint format at {ckpt_file}")

    share = bool(getattr(self.config, 'share_vae_across_layers', True))
    self.kv_vae_deterministic = bool(getattr(self.config, 'kv_vae_deterministic', True))
    self.kv_vae_apply_on_decode_only = bool(getattr(self.config, 'kv_vae_apply_on_decode_only', False))
    self.use_kv_vae = True

    if share:
        if not hasattr(self, '_shared_inference_vae'):
            # create pool locally on the attention object for simplicity
            # weights are tied by reusing the exact same submodule for each group index on each layer.
            pass
        pool = SharedInferenceVAEPool(self.config)
        group_idx = int(self.layer_idx) // int(getattr(self.config, 'vae_group_size', 4))
        _load_vae_group_state(pool.get_vae(int(self.layer_idx)), state, int(self.layer_idx), group_idx)
        self.kv_vae = pool.get_vae(int(self.layer_idx))
    else:
        kv_dim = self.num_key_value_heads * self.head_dim
        self.kv_vae = InferenceTokenVAE(
            input_dim=2 * kv_dim,
            latent_dim=int(getattr(self.config, 'kv_latent_size', 32)),
            hidden_dim=int(getattr(self.config, 'vae_hidden_size', 256)),
            logvar_min=float(getattr(self.config, 'logvar_min', -4.0)),
            logvar_max=float(getattr(self.config, 'logvar_max', 1.0)),
        )
        _load_vae_group_state(self.kv_vae, state, int(self.layer_idx), 0)

    target_dtype = self.q_proj.weight.dtype
    target_device = self.q_proj.weight.device
    self.kv_vae.to(device=target_device, dtype=target_dtype)
    self.kv_vae.eval()
    for p in self.kv_vae.parameters():
        p.requires_grad = False

    self.kv_vae_inited = True
    logger.info(
        f"Initialized KV-VAE for layer {self.layer_idx}. deterministic={self.kv_vae_deterministic}, "
        f"decode_only={self.kv_vae_apply_on_decode_only}, share={share}"
    )


def apply_kv_vae_if_needed(attn_module, key_states: torch.Tensor, value_states: torch.Tensor, q_len: int):
    """
    Input/Output shapes:
      key_states:   [B, num_kv_heads, T, Hd]   pre-RoPE
      value_states: [B, num_kv_heads, T, Hd]   pre-RoPE
    """
    init_kv_vae_for_mistral_attn(attn_module)
    if not getattr(attn_module, 'use_kv_vae', False):
        return key_states, value_states

    if getattr(attn_module, 'kv_vae_apply_on_decode_only', False) and q_len > 1:
        return key_states, value_states

    bsz, num_kv_heads, seq_len, head_dim = key_states.shape
    key_flat = key_states.transpose(1, 2).contiguous().reshape(bsz, seq_len, num_kv_heads * head_dim)
    value_flat = value_states.transpose(1, 2).contiguous().reshape(bsz, seq_len, num_kv_heads * head_dim)
    kv = torch.cat([key_flat, value_flat], dim=-1)

    with torch.no_grad():
        recon_kv, mu, logvar, z = attn_module.kv_vae(kv, deterministic=getattr(attn_module, 'kv_vae_deterministic', True))

    recon_k_flat, recon_v_flat = torch.split(recon_kv, key_flat.shape[-1], dim=-1)
    recon_k = recon_k_flat.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2).contiguous()
    recon_v = recon_v_flat.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2).contiguous()

    # optional debug stats on module
    if getattr(attn_module.config, 'kv_vae_debug', False):
        attn_module._kv_vae_last_stats = {
            'mu_mean': mu.detach().float().mean().item(),
            'mu_std': mu.detach().float().std(unbiased=False).item(),
            'logvar_mean': logvar.detach().float().mean().item(),
            'logvar_std': logvar.detach().float().std(unbiased=False).item(),
            'seq_len': seq_len,
            'q_len': q_len,
        }

    return recon_k, recon_v

# dropped token 不丢，而是存 latent(mu) + 原始位置
# prefix 阶段构建 hybrid cache
# decode 阶段恢复 dropped latent 并拼接回 attention
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
    init_snapkv(self)

    if 'padding_mask' in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )
        attention_mask = kwargs.pop('padding_mask')

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # pre-RoPE, unrepeated
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states_pre = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states_pre = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states_pre.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, 'kv_seq_len'):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
    cos, sin = self.rotary_emb(value_states_pre, seq_len=rotary_seq_len)

    # 当前 query / current prefix attention 仍然使用原始当前 KV
    query_states, key_states_post = apply_rotary_pos_emb(query_states, key_states_pre, cos, sin, position_ids)

    use_sliding_windows = (
        _flash_supports_window_size
        and getattr(self.config, 'sliding_window', None) is not None
        and kv_seq_len > self.config.sliding_window
    )

    if not _flash_supports_window_size:
        logger.warning_once(
            'The current flash attention version does not support sliding window attention, for a more memory efficient implementation'
            ' make sure to upgrade flash-attn library.'
        )

    key_states_rep = repeat_kv(key_states_post, self.num_key_value_groups)
    value_states_rep = repeat_kv(value_states_pre, self.num_key_value_groups)

    if past_key_value is not None:
        cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
        if (
            getattr(self.config, 'sliding_window', None) is not None
            and kv_seq_len > self.config.sliding_window
            and cache_has_contents
        ):
            slicing_tokens = 1 - self.config.sliding_window
            past_key = past_key_value[self.layer_idx][0]
            past_value = past_key_value[self.layer_idx][1]
            past_key = past_key[:, :, slicing_tokens:, :].contiguous()
            past_value = past_value[:, :, slicing_tokens:, :].contiguous()

            if past_key.shape[-2] != self.config.sliding_window - 1:
                raise ValueError(
                    f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got {past_key.shape}"
                )

            if attention_mask is not None:
                attention_mask = attention_mask[:, slicing_tokens:]
                attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

        cache_kwargs = {'sin': sin, 'cos': cos}

        # prefix phase
        if key_states_rep.shape[-2] >= kv_seq_len:
            self.kv_seq_len = kv_seq_len

            hybrid_k, hybrid_v, used_hybrid = _maybe_build_hybrid_cache(
                self,
                key_states_pre=key_states_pre,
                value_states_pre=value_states_pre,
                key_states_rep_postrope=key_states_rep,
                query_states_rep_postrope=query_states,
                cos=cos,
                sin=sin,
                position_ids=position_ids,
            )

            if used_hybrid:
                past_key_value.update(hybrid_k, hybrid_v, self.layer_idx, cache_kwargs)
            else:
                key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                    key_states_rep, query_states, value_states_rep, attention_mask, self.num_key_value_groups
                )
                past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)

        # decode phase
        else:
            self.kv_seq_len += q_len
            key_states_rep, value_states_rep = past_key_value.update(
                key_states_rep, value_states_rep, self.layer_idx, cache_kwargs
            )

            restored_k, restored_v = _maybe_restore_dropped_cache(self, cos, sin)
            if restored_k is not None and restored_v is not None:
                key_states_rep = torch.cat([key_states_rep, restored_k], dim=2)
                value_states_rep = torch.cat([value_states_rep, restored_v], dim=2)

    dropout_rate = 0.0 if not self.training else self.attention_dropout

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if hasattr(self.config, '_pre_quantization_dtype'):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32. We will cast back the input in {target_dtype}."
        )
        query_states = query_states.to(target_dtype)
        key_states_rep = key_states_rep.to(target_dtype)
        value_states_rep = value_states_rep.to(target_dtype)

    query_states = query_states.transpose(1, 2)
    key_states = key_states_rep.transpose(1, 2)
    value_states = value_states_rep.transpose(1, 2)

    attn_output = self._flash_attention_forward(
        query_states,
        key_states,
        value_states,
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

def prepare_inputs_for_generation_mistral(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
    if past_key_values is None:
        for layer in self.model.layers:
            layer.self_attn.kv_seq_len = 0

            # new：把latent cache也清除掉
            if hasattr(layer.self_attn, "kv_cluster"):
                layer.self_attn.kv_cluster.clear_vae_cache()
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            cache_length = past_length = self.model.layers[0].self_attn.kv_seq_len
            max_cache_length = None

        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]

        if (
            max_cache_length is not None
            and attention_mask is not None
            and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get('position_ids', None)
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]

    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {'inputs_embeds': inputs_embeds}
    else:
        model_inputs = {'input_ids': input_ids}

    model_inputs.update(
        {
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
        }
    )
    return model_inputs
