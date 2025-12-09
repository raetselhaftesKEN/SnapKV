
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

# window_size：滑动窗口大小
# max_capacity_prompt：最多缓存的 token 数（旧 token + 窗口 token）
# kernel_size：对注意力得分做 1D 池化的卷积核大小
# block_size：旧 token 合并成补偿 token 的分块大小（决定补偿 token 的粒度）
class SnapKVCluster():
    def __init__(
        self,
        window_size: int = 64,
        max_capacity_prompt: int = 256 + 64,
        kernel_size: int = 5,
        pooling: str = 'avgpool',
        block_size: int = 64,
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.block_size = max(1, block_size)

    def reset(
        self,
        window_size: int = 64,
        max_capacity_prompt: int = 256 + 64,
        kernel_size: int = 5,
        pooling: str = 'avgpool',
        block_size: int = 64,
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.block_size = max(1, block_size)

    '''
    核心函数：压缩 KV 缓存（多补偿 token 版本）
    - 旧 token 中的一部分按 score（attn_cache）单独保留；
    - 其余旧 token 按 block_size 分块，在每个块内根据注意力权重加权合并成一个补偿 token；
    - 总的 “旧 token + 补偿 token” 数量不超过 max_capacity_prompt - window_size。
    '''
    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        # 缓存没满，不压缩
        if q_len < self.max_capacity_prompt:
            return key_states, value_states

        # ===== 基本量 =====
        total_len = q_len
        window_size = self.window_size
        old_len = total_len - window_size                 # 旧 token 数量
        assert old_len > 0
        capacity_old = self.max_capacity_prompt - window_size  # 旧 token 的容量上限
        assert capacity_old > 0

        block_size = self.block_size
        n_blocks = (old_len + block_size - 1) // block_size    # 向上取整的块数

        # ===== 计算观察窗口对所有 token 的注意力 =====
        # attn_weights: [B, H, window_size, total_len]
        attn_weights = torch.matmul(
            query_states[..., -window_size:, :],
            key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)

        # 为窗口内构造下三角 mask（因果）
        mask = torch.full(
            (window_size, window_size),
            torch.finfo(attn_weights.dtype).min,
            device=attn_weights.device
        )
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        attention_mask_local = mask[None, None, :, :]

        # 加上窗口内的 mask 并 softmax
        attn_weights[:, :, -window_size:, -window_size:] += attention_mask_local
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # ===== 对旧 token 的注意力汇总，用于评分 =====
        # [B, H, old_len]
        attn_weights_sum = attn_weights[:, :, -window_size:, : -window_size].sum(dim=-2)

        # 1D 池化平滑，得到 attn_cache 作为 score
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

        # 旧 KV
        # k_old/v_old: [B, H, old_len, D]
        k_old = key_states[:, :, :-window_size, :]
        v_old = value_states[:, :, :-window_size, :]

        # ===== 计算“单独保留”的旧 token 数量 & 是否只保留补偿块 =====
        # 情况 A：capacity_old <= n_blocks 块数比容量多，极端情况
        #   - 只能保留不超过 capacity_old 个“补偿块”，不再单独保留 token
        # 情况 B：capacity_old > n_blocks
        #   - 使用所有 n_blocks 补偿块，并额外保留 keep_old_eff 个旧 token
        if capacity_old <= n_blocks:
            only_blocks = True
            keep_old_eff = 0
        else:
            only_blocks = False
            # capacity_old = keep_old_eff + n_blocks
            # 同时 old_len >= keep_old_eff + n_blocks（否则就没那么多旧 token）
            keep_old_eff = min(capacity_old - n_blocks, old_len - n_blocks)
            keep_old_eff = max(keep_old_eff, 0)

        # ===== 选出要“单独保留”的旧 token（可选）=====
        device = key_states.device
        keep_mask = torch.zeros(bsz, num_heads, old_len, dtype=torch.bool, device=device)
        if not only_blocks and keep_old_eff > 0:
            # attn_cache: [B, H, old_len]
            # top-k 作为单独保留 token
            keep_idx = attn_cache.topk(keep_old_eff, dim=-1).indices     # [B, H, keep_old_eff]
            keep_mask.scatter_(dim=-1, index=keep_idx, value=True)       # 标记为保留
        # 需要合并的旧 token mask
        merge_mask = ~keep_mask                                           # [B, H, old_len]

        # ===== 计算每个 block 的补偿 token（多补偿）=====
        # 对 merge_mask==True 的 token，在各自 block 内用 attn_weights_sum 作为权重加权平均
        k_merge_blocks = []
        v_merge_blocks = []

        eps = 1e-8
        for b in range(n_blocks):
            start = b * block_size
            end = min((b + 1) * block_size, old_len)

            # 这一块中需要合并的 token mask / 权重
            merge_mask_block = merge_mask[..., start:end]                              # [B, H, Lb]
            weights_block = attn_weights_sum[..., start:end] * merge_mask_block.to(attn_weights_sum.dtype)  # [B, H, Lb]

            sum_block = weights_block.sum(dim=-1, keepdim=True)                        # [B, H, 1]
            # 为避免 0 除，只有 sum>0 才做归一化，否则整块权重置 0（这类块整体贡献趋近于 0）
            norm_block = torch.where(
                sum_block > 0,
                weights_block / (sum_block + eps),
                torch.zeros_like(weights_block)
            )                                                                          # [B, H, Lb]

            # 该块的补偿 K/V：对该 block 内被合并的 token 做加权平均
            comp_k_block = (norm_block.unsqueeze(-1) * k_old[..., start:end, :]).sum(dim=2, keepdim=True)   # [B,H,1,D]
            comp_v_block = (norm_block.unsqueeze(-1) * v_old[..., start:end, :]).sum(dim=2, keepdim=True)

            k_merge_blocks.append(comp_k_block)
            v_merge_blocks.append(comp_v_block)

        # 所有 block 的补偿 token 堆叠
        # [B, H, n_blocks, D]
        k_merge_all = torch.cat(k_merge_blocks, dim=2)
        v_merge_all = torch.cat(v_merge_blocks, dim=2)

        # ===== 根据容量选择使用哪些 block 的补偿 token =====
        if only_blocks:
            # 只能保留不超过 capacity_old 个补偿块：根据 block 重要性（attn_cache 在该块上的总和）选 top-k
            block_scores_list = []
            for b in range(n_blocks):
                start = b * block_size
                end = min((b + 1) * block_size, old_len)
                score_block = attn_cache[..., start:end].sum(dim=-1, keepdim=True)   # [B, H, 1]
                block_scores_list.append(score_block)
            block_scores = torch.cat(block_scores_list, dim=-1)                      # [B, H, n_blocks]

            k_blocks = min(capacity_old, n_blocks)
            block_idx = block_scores.topk(k_blocks, dim=-1).indices                  # [B, H, k_blocks]
            idx_expanded = block_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)      # [B,H,k_blocks,D]

            k_blocks_sel = k_merge_all.gather(dim=2, index=idx_expanded)             # [B,H,k_blocks,D]
            v_blocks_sel = v_merge_all.gather(dim=2, index=idx_expanded)

            k_past_compress = k_blocks_sel
            v_past_compress = v_blocks_sel

        else:
            # 可以保留所有 n_blocks 的补偿块，再加 keep_old_eff 个单独旧 token
            if keep_old_eff > 0:
                keep_idx = attn_cache.topk(keep_old_eff, dim=-1).indices             # [B, H, keep_old_eff]
                keep_idx_expanded = keep_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [B,H,keep_old_eff,D]
                k_past_keep = k_old.gather(dim=2, index=keep_idx_expanded)
                v_past_keep = v_old.gather(dim=2, index=keep_idx_expanded)

                # 拼接：单独保留的旧 token + 所有 block 补偿 token
                k_past_compress = torch.cat([k_past_keep, k_merge_all], dim=2)       # [B,H,keep_old_eff + n_blocks,D]
                v_past_compress = torch.cat([v_past_keep, v_merge_all], dim=2)
            else:
                # 没有单独保留 token，只用所有 block 的补偿 token
                k_past_compress = k_merge_all
                v_past_compress = v_merge_all

        # 理论上旧 token 总数不超过 capacity_old：
        #   only_blocks:  len = k_blocks <= capacity_old
        #   else:        len = keep_old_eff + n_blocks <= capacity_old

        # ===== 拼接当前窗口 =====
        k_cur = key_states[:, :, -window_size:, :]
        v_cur = value_states[:, :, -window_size:, :]

        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)

        # 总长度约为 min(old_len, capacity_old) + window_size，不超过 max_capacity_prompt
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
        if not hasattr(self.config, 'block_size'):
            # 默认按 window_size 或固定值都可以，你可以自己调
            self.config.block_size = 64
    self.kv_cluster = SnapKVCluster(
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        kernel_size=self.config.kernel_size,
        pooling=self.config.pooling,
        block_size=self.config.block_size,
    )