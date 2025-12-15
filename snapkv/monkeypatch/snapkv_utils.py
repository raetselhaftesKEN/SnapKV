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


# window_size：滑动窗口大小  max_capacity_prompt：做多缓存token数   kernel_size：对注意力得分池化的卷积核大小
class SnapKVCluster():
    def __init__(self, window_size=64, max_capacity_prompt=256 + 64, kernel_size=5, pooling='avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(self, window_size=64, max_capacity_prompt=256 + 64, kernel_size=5, pooling='avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    '''
    核心函数：压缩kv缓存
    整个kv缓存的操作都在这里面
    你要加入token补偿这些的，就也在这里面加，应该核心向量是attn_weights

    加入了补偿token：每次压缩生成一个单独的补偿token，然后加到max_capacity_prompt里面
    '''

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        if q_len < self.max_capacity_prompt:  # 缓存没满，不压缩
            return key_states, value_states
        else:
            # 当前序列长度
            total_len = q_len
            # 旧 token 数量（不含滑动窗口）
            old_len = total_len - self.window_size
            # 旧 token 能保留的总容量
            capacity_old = self.max_capacity_prompt - self.window_size
            assert capacity_old > 0
            # 预留 1 个位置给补偿 token，其余 capacity_old-1 个位置给“单独保留”的旧token
            keep_old = max(capacity_old - 1, 0)

            # attn_weights：只看window_size个token
            # 下面这一行就在计算观察窗口对全部token的注意力
            attn_weights = torch.matmul(
                query_states[..., -self.window_size:, :],
                key_states.transpose(2, 3)
            ) / math.sqrt(head_dim)

            # 然后计算观察窗口内部的注意力（因果mask）
            mask = torch.full(
                (self.window_size, self.window_size),
                torch.finfo(attn_weights.dtype).min,
                device=attn_weights.device
            )
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            # 两个注意力拼起来
            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask
            # softmax正则化
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # 对旧的token计算观察窗口的注意力之和（用于舍弃/保留评分）
            # 形状: [B, H, old_len]
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim=-2)

            # 下面是做一个1D池化平滑，防止单点噪声
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(
                    attn_weights_sum,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    stride=1
                )
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(
                    attn_weights_sum,
                    kernel_size=self.kernel_size,
                    padding=self.kernel_size // 2,
                    stride=1
                )
            else:
                raise ValueError('Pooling method not supported')

            # ============ 新增：token merging 补偿 ============

            # 对旧 token 做 top-k，先选出 capacity_old 个候选位置
            # 其中前 keep_old 个单独保留，剩余的 + 其余未进 top-k 的统统合并成一个补偿 token
            capacity_old_eff = min(capacity_old, old_len)  # 防御性，通常 old_len >= capacity_old
            keep_old_eff = max(capacity_old_eff - 1, 0)

            topk_vals, topk_idx = attn_cache.topk(capacity_old_eff, dim=-1)  # [B, H, capacity_old_eff]

            # 单独保留的旧 token 索引：[B, H, keep_old_eff]
            if keep_old_eff > 0:
                keep_idx = topk_idx[..., :keep_old_eff]
            else:
                # keep_old_eff == 0 时，没有单独保留的旧 token
                keep_idx = None

            # 构造 mask，标记哪些旧 token 被单独保留
            device = key_states.device
            keep_mask = torch.zeros(bsz, num_heads, old_len, dtype=torch.bool, device=device)
            if keep_old_eff > 0:
                keep_mask.scatter_(
                    dim=-1,
                    index=keep_idx,
                    value=True
                )

            # merge_mask: 需要合并进“补偿 token”的所有旧 token
            merge_mask = ~keep_mask  # [B, H, old_len]

            # 注意力权重作为 merge 权重（对被合并 token 做加权平均）
            merge_weights = attn_weights_sum * merge_mask.to(attn_weights_sum.dtype)  # [B, H, old_len]
            # 防止全 0
            eps = 1e-8
            merge_weights_sum = merge_weights.sum(dim=-1, keepdim=True) + eps
            merge_weights_norm = merge_weights / merge_weights_sum  # 归一化

            # 旧的 K/V
            k_old = key_states[:, :, :-self.window_size, :]  # [B, H, old_len, D]
            v_old = value_states[:, :, :-self.window_size, :]

            # 计算补偿 token 的 K/V：对需要合并的 token 做加权平均
            # 形状：[B, H, 1, D]
            k_merge = (merge_weights_norm.unsqueeze(-1) * k_old).sum(dim=2, keepdim=True)
            v_merge = (merge_weights_norm.unsqueeze(-1) * v_old).sum(dim=2, keepdim=True)

            # 选出单独保留的旧 token 的 K/V
            if keep_old_eff > 0:
                keep_idx_expanded = keep_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [B, H, keep_old_eff, D]
                k_past_keep = k_old.gather(dim=2, index=keep_idx_expanded)
                v_past_keep = v_old.gather(dim=2, index=keep_idx_expanded)
                # 拼成新的“旧 token”KV：单独保留的 + 1 个补偿 token
                k_past_compress = torch.cat([k_past_keep, k_merge], dim=2)
                v_past_compress = torch.cat([v_past_keep, v_merge], dim=2)
            else:
                # capacity_old_eff == 1 且 keep_old_eff == 0 的极端情况：
                # 只用一个补偿 token 表示所有旧 token
                k_past_compress = k_merge
                v_past_compress = v_merge

            # 理论上 k_past_compress 的长度应为 capacity_old_eff（<= capacity_old）
            # 如果 old_len < capacity_old，说明旧 token 本来就少，直接保留所有旧 token + 补偿也不会超过限制
            # 下游只看总长度不超过 max_capacity_prompt 即可

            # ============ 原有逻辑：拼接当前窗口 ============

            # 然后拼接 “压缩后的旧 token KV” 和 “观察窗口的 KV 缓存”
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)

            # 最终 key_states/value_states 长度约为：min(old_len, capacity_old) + window_size
            # 一般情况等于 max_capacity_prompt，不超过原设定容量
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