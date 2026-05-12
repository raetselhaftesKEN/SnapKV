import inspect
import json
import os
import math
import warnings
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
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)


# ============================================================
# Reconstruction-first KV-VAE inference helpers
# ============================================================
def rms_normalize_last_dim(x: torch.Tensor, eps: float = 1e-6):
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return x / rms, rms


class InferenceHeadwiseVAECore(nn.Module):
    """
    Mirror of train_mistral_kv_recon_vae.py -> HeadwiseVAECore
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, logvar_min: float = -8.0, logvar_max: float = -2.0):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        self.enc_in = nn.Linear(input_dim, hidden_dim)
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

    def encode(self, x: torch.Tensor):
        x_norm, x_rms = rms_normalize_last_dim(x)

        h = F.silu(self.enc_in(x_norm))
        h_res = h
        h = F.silu(self.enc_fc1(h))
        h = self.enc_fc2(h)
        h = self.enc_ln(h + h_res)
        h = F.silu(h)

        mu = self.mu_proj(h)
        logvar_raw = self.logvar_proj(h)
        logvar = torch.clamp(logvar_raw, min=self.logvar_min, max=self.logvar_max)
        return mu, logvar, x_rms

    def decode(self, z: torch.Tensor, x_rms: torch.Tensor):
        h = F.silu(self.dec_in(z))
        h_res = h
        h = F.silu(self.dec_fc1(h))
        h = self.dec_fc2(h)
        h = self.dec_ln(h + h_res)
        h = F.silu(h)
        out_norm = self.out_proj(h)
        out = out_norm * x_rms
        return out

    def forward(self, x: torch.Tensor, deterministic: bool = True):
        mu, logvar, x_rms = self.encode(x)
        if deterministic:
            z = mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        recon = self.decode(z, x_rms)
        return recon, mu, logvar, z


class InferenceLayerwiseKVVAE(nn.Module):
    """
    Mirror of train_mistral_kv_recon_vae.py -> LayerwiseKVVAE
    - split K / V
    - compress each kv-head independently
    """
    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        per_head_latent_size: int,
        hidden_dim: int,
        logvar_min: float = -8.0,
        logvar_max: float = -2.0,
        chunk_size: int = 1024,
    ):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.per_head_latent_size = per_head_latent_size
        self.chunk_size = chunk_size

        self.k_core = InferenceHeadwiseVAECore(
            input_dim=head_dim,
            latent_dim=per_head_latent_size,
            hidden_dim=hidden_dim,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        )
        self.v_core = InferenceHeadwiseVAECore(
            input_dim=head_dim,
            latent_dim=per_head_latent_size,
            hidden_dim=hidden_dim,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        )

    def _run_core(self, x_flat: torch.Tensor, core: nn.Module, deterministic: bool):
        # x_flat: [B, T, num_kv_heads * head_dim]
        bsz, seq_len, total_dim = x_flat.shape
        x = x_flat.view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        x = x.reshape(bsz * seq_len * self.num_kv_heads, self.head_dim)

        recons = []
        mus = []
        logvars = []

        for start in range(0, x.size(0), self.chunk_size):
            end = min(start + self.chunk_size, x.size(0))
            x_chunk = x[start:end]
            recon, mu, logvar, _ = core(x_chunk, deterministic=deterministic)
            recons.append(recon)
            mus.append(mu)
            logvars.append(logvar)

        recon = torch.cat(recons, dim=0)
        mu = torch.cat(mus, dim=0)
        logvar = torch.cat(logvars, dim=0)

        recon = recon.view(bsz, seq_len, self.num_kv_heads, self.head_dim).reshape(bsz, seq_len, total_dim)
        mu = mu.view(bsz, seq_len, self.num_kv_heads * self.per_head_latent_size)
        logvar = logvar.view(bsz, seq_len, self.num_kv_heads * self.per_head_latent_size)
        return recon, mu, logvar

    def forward(self, key_states_flat: torch.Tensor, value_states_flat: torch.Tensor, deterministic: bool = True):
        k_recon, k_mu, k_logvar = self._run_core(key_states_flat, self.k_core, deterministic=deterministic)
        v_recon, v_mu, v_logvar = self._run_core(value_states_flat, self.v_core, deterministic=deterministic)

        mu = torch.cat([k_mu, v_mu], dim=-1)
        logvar = torch.cat([k_logvar, v_logvar], dim=-1)
        return k_recon, v_recon, mu, logvar


class SharedInferenceVAEPool(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.group_size = int(getattr(config, "vae_group_size", 1))
        self.num_layers = int(config.num_hidden_layers)
        self.num_groups = math.ceil(self.num_layers / self.group_size)

        head_dim = config.hidden_size // config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        chunk_size = int(getattr(config, "head_chunk_size", 1024))

        self.group_vaes = nn.ModuleList([
            InferenceLayerwiseKVVAE(
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                per_head_latent_size=int(getattr(config, "per_head_latent_size", 32)),
                hidden_dim=int(getattr(config, "vae_hidden_size", 256)),
                logvar_min=float(getattr(config, "logvar_min", -8.0)),
                logvar_max=float(getattr(config, "logvar_max", -2.0)),
                chunk_size=chunk_size,
            )
            for _ in range(self.num_groups)
        ])

    def get_vae(self, layer_idx: int) -> nn.Module:
        return self.group_vaes[layer_idx // self.group_size]


def _strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return {}
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _load_vae_group_state(
    vae_module: nn.Module,
    full_state: Dict[str, torch.Tensor],
    layer_idx: int,
    group_idx: int,
    group_size: int = 1,
):
    group_start_layer = group_idx * group_size

    candidates = [
        f"model.shared_vae_pool.group_vaes.{group_idx}.",
        f"shared_vae_pool.group_vaes.{group_idx}.",
        f"model.layers.{layer_idx}.self_attn.kv_vae.",
        f"layers.{layer_idx}.self_attn.kv_vae.",
        f"model.layers.{group_start_layer}.self_attn.kv_vae.",
        f"layers.{group_start_layer}.self_attn.kv_vae.",
    ]

    loaded = False
    for prefix in candidates:
        sub = _strip_prefix_if_present(full_state, prefix)
        if sub:
            missing, unexpected = vae_module.load_state_dict(sub, strict=False)
            logger.info(
                f"Loaded VAE weights for layer={layer_idx}, group={group_idx} "
                f"from prefix='{prefix}', missing={len(missing)}, unexpected={len(unexpected)}"
            )
            loaded = True
            break

    if not loaded:
        preview_keys = list(full_state.keys())[:60]
        raise KeyError(
            f"Could not find VAE weights for layer {layer_idx} / group {group_idx}. "
            f"Tried prefixes: {candidates}. "
            f"First checkpoint keys: {preview_keys}"
        )


def init_kv_vae_for_mistral_attn(self):
    use_kv_vae = bool(getattr(self.config, "use_kv_vae", False))
    if not use_kv_vae:
        self.use_kv_vae = False
        return

    if getattr(self, "kv_vae_inited", False):
        return

    ckpt_path = getattr(self.config, "vae_ckpt_path", None)
    if ckpt_path is None:
        raise ValueError("config.use_kv_vae=True but config.vae_ckpt_path is not set.")

    ckpt_path = str(ckpt_path)
    if os.path.isdir(ckpt_path):
        candidate = os.path.join(ckpt_path, "trainable_vae_only.bin")
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Could not find trainable_vae_only.bin under directory: {ckpt_path}")
        ckpt_file = candidate
    else:
        ckpt_file = ckpt_path

    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"VAE checkpoint not found: {ckpt_file}")

    extra_cfg_path = os.path.join(os.path.dirname(ckpt_file), "extra_vae_config.json")
    if os.path.exists(extra_cfg_path):
        with open(extra_cfg_path, "r", encoding="utf-8") as f:
            extra_cfg = json.load(f)
        for k, v in extra_cfg.items():
            if not hasattr(self.config, k):
                setattr(self.config, k, v)

    state = torch.load(ckpt_file, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint format at {ckpt_file}")

    if getattr(self.config, "kv_vae_debug", False):
        print(f"[KV-VAE DEBUG] checkpoint file: {ckpt_file}")
        print("[KV-VAE DEBUG] first 40 keys:")
        for k in list(state.keys())[:40]:
            print("   ", k)

    share = bool(getattr(self.config, "share_vae_across_layers", False))
    self.kv_vae_deterministic = bool(getattr(self.config, "kv_vae_deterministic", True))
    self.kv_vae_apply_on_decode_only = bool(getattr(self.config, "kv_vae_apply_on_decode_only", False))
    self.use_kv_vae = True

    if share:
        pool = SharedInferenceVAEPool(self.config)
        group_size = int(getattr(self.config, "vae_group_size", 1))
        group_idx = int(self.layer_idx) // group_size
        vae_module = pool.get_vae(int(self.layer_idx))

        _load_vae_group_state(
            vae_module=vae_module,
            full_state=state,
            layer_idx=int(self.layer_idx),
            group_idx=group_idx,
            group_size=group_size,
        )
        self.kv_vae = vae_module
    else:
        self.kv_vae = InferenceLayerwiseKVVAE(
            num_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            per_head_latent_size=int(getattr(self.config, "per_head_latent_size", 32)),
            hidden_dim=int(getattr(self.config, "vae_hidden_size", 256)),
            logvar_min=float(getattr(self.config, "logvar_min", -8.0)),
            logvar_max=float(getattr(self.config, "logvar_max", -2.0)),
            chunk_size=int(getattr(self.config, "head_chunk_size", 1024)),
        )
        _load_vae_group_state(
            vae_module=self.kv_vae,
            full_state=state,
            layer_idx=int(self.layer_idx),
            group_idx=0,
            group_size=1,
        )

    target_dtype = self.q_proj.weight.dtype
    target_device = self.q_proj.weight.device
    self.kv_vae.to(device=target_device, dtype=target_dtype)
    self.kv_vae.eval()
    for p in self.kv_vae.parameters():
        p.requires_grad = False

    self.kv_vae_inited = True
    logger.info(
        f"Initialized reconstruction-first KV-VAE for layer {self.layer_idx}. "
        f"deterministic={self.kv_vae_deterministic}, "
        f"decode_only={self.kv_vae_apply_on_decode_only}, share={share}"
    )


def apply_kv_vae_if_needed(attn_module, key_states: torch.Tensor, value_states: torch.Tensor, q_len: int):
    """
    key_states/value_states: [B, num_kv_heads, T, Hd], pre-RoPE
    """
    init_kv_vae_for_mistral_attn(attn_module)
    if not getattr(attn_module, "use_kv_vae", False):
        return key_states, value_states

    if getattr(attn_module, "kv_vae_apply_on_decode_only", False) and q_len > 1:
        return key_states, value_states

    bsz, num_kv_heads, seq_len, head_dim = key_states.shape

    key_flat = key_states.transpose(1, 2).contiguous().reshape(bsz, seq_len, num_kv_heads * head_dim)
    value_flat = value_states.transpose(1, 2).contiguous().reshape(bsz, seq_len, num_kv_heads * head_dim)

    with torch.no_grad():
        recon_k_flat, recon_v_flat, mu, logvar = attn_module.kv_vae(
            key_flat,
            value_flat,
            deterministic=getattr(attn_module, "kv_vae_deterministic", True),
        )

    recon_k = recon_k_flat.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2).contiguous()
    recon_v = recon_v_flat.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2).contiguous()

    if getattr(attn_module.config, "kv_vae_debug", False):
        attn_module._kv_vae_last_stats = {
            "mu_mean": mu.detach().float().mean().item(),
            "mu_std": mu.detach().float().std(unbiased=False).item(),
            "logvar_mean": logvar.detach().float().mean().item(),
            "logvar_std": logvar.detach().float().std(unbiased=False).item(),
            "seq_len": seq_len,
            "q_len": q_len,
        }

    return recon_k, recon_v


def mistral_flash_attn2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    init_snapkv(self)

    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead."
        )
        attention_mask = kwargs.pop("padding_mask")

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # reconstruction-first VAE on ALL pre-RoPE KV
    key_states, value_states = apply_kv_vae_if_needed(self, key_states, value_states, q_len)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
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
            "The current flash attention version does not support sliding window attention, "
            "for a more memory efficient implementation make sure to upgrade flash-attn library."
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
        if (
            getattr(self.config, "sliding_window", None) is not None
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

        cache_kwargs = {"sin": sin, "cos": cos}
        if key_states.shape[-2] >= kv_seq_len:
            self.kv_seq_len = kv_seq_len
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask, self.num_key_value_groups
            )
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    dropout_rate = 0.0 if not self.training else self.attention_dropout

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32. We will cast back the input in {target_dtype}."
        )
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    attn_output = self._flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
        use_sliding_windows=use_sliding_windows,
    )

    try:
        comp_ok = (
            past_key_value is not None
            and getattr(self, "kv_cluster", None) is not None
            and getattr(self.kv_cluster, "comp_enabled", False)
            and getattr(self.kv_cluster, "_comp_inited", False)
            and q_len == 1
        )
    except Exception:
        comp_ok = False

    if comp_ok:
        query_states_bhld = query_states.transpose(1, 2)
        key_states_bhkd = key_states.transpose(1, 2)

        q_bhld = query_states_bhld.to(torch.float32)
        k_bhkd = key_states_bhkd.to(torch.float32)
        logits = torch.matmul(q_bhld, k_bhkd.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                key_mask = attention_mask[:, None, None, :].to(torch.float32)
                logits = logits + (1.0 - key_mask) * torch.finfo(logits.dtype).min
            elif attention_mask.dim() == 4:
                logits = logits + attention_mask.to(logits.dtype)

        m = logits.max(dim=-1).values
        exp_logits = torch.exp(logits - m.unsqueeze(-1))
        Z_Cs = exp_logits.sum(dim=-1)
        O_C = attn_output.permute(0, 2, 1, 3).to(torch.float32)
        N_Cs = Z_Cs.unsqueeze(-1) * O_C

        Z_D_raw, N_D_raw = self.kv_cluster.comp_terms(query_states_bhld)
        if (Z_D_raw is not None) and (N_D_raw is not None):
            scale = torch.exp(-m)
            Z_Ds = scale * Z_D_raw.to(torch.float32)
            N_Ds = scale.unsqueeze(-1) * N_D_raw.to(torch.float32)
            denom = (Z_Cs + Z_Ds).clamp_min(1e-6)
            O_hat = (N_Cs + N_Ds) / denom.unsqueeze(-1)
            attn_output = O_hat.permute(0, 2, 1, 3).to(attn_output.dtype)

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    attn_weights = None if not output_attentions else None
    return attn_output, attn_weights, past_key_value


def prepare_inputs_for_generation_mistral(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    **kwargs,
):
    if past_key_values is None:
        for layer in self.model.layers:
            layer.self_attn.kv_seq_len = 0

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

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]

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