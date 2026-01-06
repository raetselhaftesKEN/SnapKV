
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math

# perform qk calculation and get indices
# this version will not update in inference mode


import torch
import math

@torch.no_grad()
def snapkv_merge_discarded_tokens_cam_style(
    value_states: torch.Tensor,        # [bsz, heads, seq_len, head_dim]
    attn_weights: torch.Tensor,         # [bsz, heads, window_size, seq_len]  (softmax后)
    keep_indices_old: torch.Tensor,     # [bsz, heads, K]  (SnapKV选出的旧token下标，范围[0, old_len))
    old_len: int,                       # old_len = seq_len - window_size
    recent_budget: int                  # recent_budget = window_size
) -> torch.Tensor:
    """
    将 SnapKV 将要舍弃的旧 token（0..old_len-1 中未被 keep 的）按 LlamaAttention_cam.local_cam_mask 的规则
    把 V[p] 以 Bernoulli 门控后均分加到 V[p+1 : p+recent_budget+1] 上。
    """
    bsz, nheads, seq_len, head_dim = value_states.shape
    assert attn_weights.shape[-1] == seq_len
    assert recent_budget > 0
    merge_budget = recent_budget  # 与 local_cam_mask 一致

    # 1) CAM: attn_score = mean(attn_weights[..., :token_index, :token_index], dim=-2)
    # 在 SnapKV 里我们只有最后 window_size 个 query 的权重，所以用这 window 上的 mean（与 CAM “对query维取均值”同型）
    # attn_score: [bsz, heads, seq_len]
    attn_score = attn_weights.mean(dim=-2)

    # 2) 计算“start区域”的 max：这里用 SnapKV 保留的旧 token 集合代替 CAM 的 [:start_budget]
    # kept_old_attn: [bsz, heads, K]
    if keep_indices_old.numel() > 0:
        kept_old_attn = attn_score[:, :, :old_len].gather(dim=-1, index=keep_indices_old)
        start_max = kept_old_attn.max(dim=-1).values  # [bsz, heads]
    else:
        start_max = torch.zeros((bsz, nheads), device=value_states.device, dtype=attn_score.dtype)

    # 3) 构造旧token是否保留的mask（按 head 分开）
    keep_mask_old = torch.zeros((bsz, nheads, old_len), device=value_states.device, dtype=torch.bool)
    if keep_indices_old.numel() > 0:
        keep_mask_old.scatter_(dim=-1, index=keep_indices_old.clamp(0, old_len - 1), value=True)

    # 4) 对每个旧token位置 p：若 p 被舍弃，则执行 CAM 合并
    # CAM 中 source = token_index - recent_budget；目标区间是 source+1 : source+merge_budget+1
    for p in range(old_len):
        # 只合并被舍弃的旧token
        discard = ~keep_mask_old[:, :, p]  # [bsz, heads]
        if not discard.any():
            continue

        # window_max 对应 CAM 的 attn_score[..., p : p+recent_budget] 的 max
        # （CAM 用 torch.max(cat(...), dim=-1)[0]，这里拆成 start_max 与 window_max 的 max，完全等价）
        window_slice = attn_score[:, :, p : p + recent_budget]  # [bsz, heads, recent_budget]
        window_max = window_slice.max(dim=-1).values            # [bsz, heads]
        mean_attn = torch.maximum(start_max, window_max).clamp_min(1e-12)

        merge_prob = (attn_score[:, :, p] / mean_attn).clamp(0.0, 1.0)  # [bsz, heads]
        # 只对 discard 的位置采样，其余强制为0
        merge_prob = merge_prob * discard.to(merge_prob.dtype)

        merge_mask = torch.bernoulli(merge_prob)  # [bsz, heads] 0/1
        score1 = value_states[:, :, p, :] * merge_mask.unsqueeze(-1) / merge_budget  # [bsz, heads, head_dim]

        dst_start = p + 1
        dst_end = p + merge_budget + 1  # 刚好 recent_budget 个token
        # 这里 dst_end <= old_len + recent_budget == seq_len，永不越界
        value_states[:, :, dst_start:dst_end, :] += score1.unsqueeze(2)

    return value_states


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


# window_size：滑动窗口大小  max_capacity_prompt：做多缓存token数   kernel_size：对注意力得分池化的卷积核大小
class SnapKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    '''
    核心函数：压缩kv缓存
    整个kv缓存的操作都在这里面
    你要加入token补偿这些的，就也在这里面加，应该核心向量是attn_weights
    '''

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        if q_len < self.max_capacity_prompt:
            return key_states, value_states

        # 1) 计算窗口 query 对全量 key 的注意力（与你原逻辑一致）
        attn_weights = torch.matmul(
            query_states[..., -self.window_size:, :],
            key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)

        # 2) 窗口内部因果mask（与你原逻辑一致，但别覆盖入参attention_mask命名以免混淆）
        mask = torch.full((self.window_size, self.window_size),
                          torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        window_attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:, -self.window_size:] += window_attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # 3) 旧token重要性得分（与你原逻辑一致：sum over window queries）
        attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim=-2)

        if self.pooling == 'avgpool':
            attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=self.kernel_size,
                                      padding=self.kernel_size // 2, stride=1)
        elif self.pooling == 'maxpool':
            attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=self.kernel_size,
                                      padding=self.kernel_size // 2, stride=1)
        else:
            raise ValueError('Pooling method not supported')

        # 4) 选出要保留的旧token（raw indices，后面合并要用）
        K = self.max_capacity_prompt - self.window_size
        indices_1d = attn_cache.topk(K, dim=-1).indices  # [bsz, heads, K]  值域[0, old_len)

        # ========= 新增：把“将要舍弃的旧token”按 CAM 数学规则合并进 value_states =========
        old_len = q_len - self.window_size
        value_states = snapkv_merge_discarded_tokens_cam_style(
            value_states=value_states,
            attn_weights=attn_weights,  # softmax后的权重
            keep_indices_old=indices_1d,
            old_len=old_len,
            recent_budget=self.window_size
        )
        # ========= 新增结束 =========

        # 5) 按 SnapKV 原逻辑 gather 压缩 KV（用 expanded indices）
        indices = indices_1d.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
        v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)

        k_cur = key_states[:, :, -self.window_size:, :]
        v_cur = value_states[:, :, -self.window_size:, :]
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
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling
        )