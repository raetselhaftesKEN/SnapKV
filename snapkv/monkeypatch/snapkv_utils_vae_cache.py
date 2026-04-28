import torch
import torch.nn.functional as F
import torch.nn as nn
import math

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class SnapKVCluster:
    def __init__(self, window_size=64, max_capacity_prompt=256 + 64, kernel_size=5, pooling='avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.clear_vae_cache()

    def reset(self, window_size=64, max_capacity_prompt=256 + 64, kernel_size=5, pooling='avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.clear_vae_cache()

    def clear_vae_cache(self):
        self.vae_dropped_latent = None
        self.vae_dropped_positions = None
        self.vae_num_kv_heads = None
        self.vae_head_dim = None
        self.vae_group_size = None
        self.vae_dropped_count = 0

    def has_vae_cache(self):
        return self.vae_dropped_latent is not None and self.vae_dropped_positions is not None

    def store_vae_compressed(self, latent, positions, num_kv_heads, head_dim, group_size):
        self.vae_dropped_latent = latent.detach()
        self.vae_dropped_positions = positions.detach()
        self.vae_num_kv_heads = int(num_kv_heads)
        self.vae_head_dim = int(head_dim)
        self.vae_group_size = int(group_size)
        self.vae_dropped_count = int(positions.shape[-1]) if positions is not None else 0

    def _compute_attn_cache(self, key_states_rep, query_states_rep):
        """
        key_states_rep/query_states_rep: [B, num_heads, T, Hd], post-RoPE, repeated KV heads
        返回 old tokens 的共享重要性分数: [B, T_old]
        """
        _, _, q_len, head_dim = query_states_rep.shape

        attn_weights = torch.matmul(
            query_states_rep[..., -self.window_size:, :],
            key_states_rep.transpose(2, 3)
        ) / math.sqrt(head_dim)

        mask = torch.full(
            (self.window_size, self.window_size),
            torch.finfo(attn_weights.dtype).min,
            device=attn_weights.device
        )
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states_rep.dtype)

        # 原始 SnapKV 是 per-head，这里聚合成共享 token ranking
        attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim=-2)  # [B,H,T_old]
        attn_weights_sum = attn_weights_sum.mean(dim=1)  # [B,T_old]

        if self.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(
                attn_weights_sum.unsqueeze(1),
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1
            ).squeeze(1)
        elif self.pooling == 'maxpool':
            attn_cache = F.max_pool1d(
                attn_weights_sum.unsqueeze(1),
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1
            ).squeeze(1)
        else:
            raise ValueError('Pooling method not supported')

        return attn_cache

    def select_important_tokens(self, key_states_rep, query_states_rep):
        """
        key_states_rep/query_states_rep: [B, num_heads, T, Hd]
        返回 old prefix token 的 keep/drop 索引
        keep_idx: [B, K]
        drop_idx: [B, T_old-K]
        """
        assert key_states_rep.shape[-2] == query_states_rep.shape[-2]
        _, _, q_len, _ = query_states_rep.shape
        old_len = q_len - self.window_size
        keep_n = self.max_capacity_prompt - self.window_size

        if old_len <= 0 or q_len < self.max_capacity_prompt:
            return None, None

        attn_cache = self._compute_attn_cache(key_states_rep, query_states_rep)
        keep_idx = attn_cache.topk(keep_n, dim=-1).indices  # [B, keep_n]

        bsz = keep_idx.size(0)
        full_idx = torch.arange(old_len, device=keep_idx.device).unsqueeze(0).expand(bsz, -1)
        keep_mask = torch.zeros(bsz, old_len, dtype=torch.bool, device=keep_idx.device)
        keep_mask.scatter_(1, keep_idx, True)
        drop_idx = full_idx.masked_select(~keep_mask).view(bsz, old_len - keep_n)

        return keep_idx, drop_idx

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        """
        原始 SnapKV fallback 逻辑，保持不动
        输入是 post-RoPE 且 repeat 后的 KV: [B, H, T, D]
        """
        assert key_states.shape[-2] == query_states.shape[-2]
        _, _, q_len, head_dim = query_states.shape
        if q_len < self.max_capacity_prompt:
            return key_states, value_states

        attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)

        mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim=-2)

        if self.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        elif self.pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        else:
            raise ValueError('Pooling method not supported')

        indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
        v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)

        k_cur = key_states[:, :, -self.window_size:, :]
        v_cur = value_states[:, :, -self.window_size:, :]
        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)
        return key_states, value_states


def init_snapkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
    self.kv_cluster = SnapKVCluster(
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        kernel_size=self.config.kernel_size,
        pooling=self.config.pooling
    )