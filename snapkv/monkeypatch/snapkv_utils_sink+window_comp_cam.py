
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math

# perform qk calculation and get indices
# this version will not update in inference mode

# Copied from transformers.models.llama.modeling_llama.repeat_kv
# 复制KV头，让KV头数和注意力对其（非核心，只是为了能跑）
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def _scatter_add_4d(dst, index, src):
    """
    dst:  [B, H, K, D]
    index:[B, H, N]  each in [0, K-1]
    src:  [B, H, N, D]
    """
    B, H, K, D = dst.shape
    # expand index to match last dim
    index_exp = index.unsqueeze(-1).expand(B, H, index.shape[-1], D)  # [B,H,N,D]
    return dst.scatter_add(2, index_exp, src)


def _scatter_add_3d(dst, index, src):
    """
    dst:  [B, H, K]
    index:[B, H, N]
    src:  [B, H, N]
    """
    return dst.scatter_add(2, index, src)


def _merge_dropped_into_kept(
    k_old: torch.Tensor,          # [B,H,L_old,D]
    v_old: torch.Tensor,          # [B,H,L_old,D]
    kept_indices: torch.Tensor,   # [B,H,K]  (topK positions in [0, L_old-1])
    weights_m: torch.Tensor,      # [B,H,L_old]  importance mass for each old token
    eps: float = 1e-6,
    merge_k: bool = False,
):
    """
    将所有 old token（包括 kept + dropped）按 key 相似度分配到 kept slots，
    然后用 weights_m 做加权平均，得到每个 kept slot 的 merged V（以及可选 merged K）。

    返回:
      k_kept_out: [B,H,K,D]
      v_kept_out: [B,H,K,D]
    """
    B, H, L_old, D = k_old.shape
    K = kept_indices.shape[-1]

    # 1) gather kept K/V 作为“代表点”
    kept_idx_exp = kept_indices.unsqueeze(-1).expand(B, H, K, D)  # [B,H,K,D]
    k_kept = k_old.gather(dim=2, index=kept_idx_exp)              # [B,H,K,D]
    v_kept = v_old.gather(dim=2, index=kept_idx_exp)              # [B,H,K,D]

    # 2) 对每个 old token 计算它与各 kept key 的相似度，选择最相似 kept slot
    # sim: [B,H,L_old,K]
    sim = torch.matmul(k_old, k_kept.transpose(-1, -2))  # dot product
    assign = sim.argmax(dim=-1)                          # [B,H,L_old] in [0, K-1]

    # 3) 以 weights_m 作为“注意力质量/重要性”做加权合并
    m = weights_m.clamp_min(0.0)                         # [B,H,L_old]
    m_unsq = m.unsqueeze(-1)                             # [B,H,L_old,1]

    # numV[k] = sum_{t assigned->k} m_t * v_t
    numV = torch.zeros((B, H, K, D), device=v_old.device, dtype=v_old.dtype)
    numV = _scatter_add_4d(numV, assign, v_old * m_unsq)

    # denom[k] = sum_{t assigned->k} m_t
    denom = torch.zeros((B, H, K), device=v_old.device, dtype=v_old.dtype)
    denom = _scatter_add_3d(denom, assign, m)

    v_merged = numV / (denom.unsqueeze(-1) + eps)         # [B,H,K,D]

    if not merge_k:
        # 默认：不改 key，避免注意力定位漂移
        return k_kept, v_merged

    # 可选：也合并 key（更激进，可能更省信息但也更不稳定）
    numK = torch.zeros((B, H, K, D), device=k_old.device, dtype=k_old.dtype)
    numK = _scatter_add_4d(numK, assign, k_old * m_unsq)
    k_merged = numK / (denom.unsqueeze(-1) + eps)

    # 让 key 长度尺度别爆：可选做归一化（更像“方向平均”）
    k_merged = F.normalize(k_merged, dim=-1)

    return k_merged, v_merged


class SnapKVCluster():
    def __init__(self, window_size=64, max_capacity_prompt=256+64, kernel_size=5, pooling='avgpool',
                 merge_dropped=True, merge_k=False):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

        # 新增：是否对 dropped tokens 做合并补偿
        self.merge_dropped = merge_dropped
        self.merge_k = merge_k  # 是否连 K 也合并（默认 False）

    def reset(self, window_size=64, max_capacity_prompt=256+64, kernel_size=5, pooling='avgpool',
              merge_dropped=True, merge_k=False):
        self.__init__(window_size, max_capacity_prompt, kernel_size, pooling, merge_dropped, merge_k)

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        if q_len < self.max_capacity_prompt:
            return key_states, value_states

        # 1) 只用窗口 query 计算对全量 key 的注意力
        attn_weights = torch.matmul(
            query_states[..., -self.window_size:, :],
            key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)

        # 2) 窗口内部 causal mask
        mask = torch.full((self.window_size, self.window_size),
                          torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        attention_mask_local = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask_local
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # 3) 旧 token 的注意力质量（mass）：m_t
        # shape: [B,H,L_old] where L_old = q_len - window_size
        attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim=-2)

        # 4) 平滑（与原 SnapKV 保持一致）
        if self.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=self.kernel_size,
                                      padding=self.kernel_size//2, stride=1)
        elif self.pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=self.kernel_size,
                                      padding=self.kernel_size//2, stride=1)
        else:
            raise ValueError('Pooling method not supported')

        # 5) topK 选择（与原 SnapKV 完全一致）
        K_keep = self.max_capacity_prompt - self.window_size
        kept_pos = attn_cache.topk(K_keep, dim=-1).indices  # [B,H,K_keep] in [0, L_old-1]

        # 6) 取出 old 与 cur
        k_old = key_states[:, :, :-self.window_size, :]     # [B,H,L_old,D]
        v_old = value_states[:, :, :-self.window_size, :]   # [B,H,L_old,D]
        k_cur = key_states[:, :, -self.window_size:, :]
        v_cur = value_states[:, :, -self.window_size:, :]

        # 7) 合并补偿：把 dropped 的信息融合到 kept slots 的 V（可选也融合 K）
        if self.merge_dropped:
            # 你可以在这里选择用 attn_cache 或 attn_weights_sum 做权重
            # 建议默认用 attn_cache（平滑后更稳）
            weights_m = attn_cache  # [B,H,L_old]
            k_past_compress, v_past_compress = _merge_dropped_into_kept(
                k_old, v_old, kept_pos, weights_m,
                eps=1e-6,
                merge_k=self.merge_k
            )
        else:
            # 原版：直接 gather
            kept_pos_exp = kept_pos.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = k_old.gather(dim=2, index=kept_pos_exp)
            v_past_compress = v_old.gather(dim=2, index=kept_pos_exp)

        # 8) 拼接 topK(old) + window(cur)
        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)
        return key_states, value_states


# 工具类，把snapKV注入注意力中，非核心算法，应该不用改
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