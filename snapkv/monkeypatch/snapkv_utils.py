
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

class SnapKVCluster:
    """
    sink + window + merge-compensation:
    - 保留最前 sink_size 个 token（sink）
    - 保留最后 window_size 个 token（recent window）
    - 中间被舍弃的 token：按“最近 window 对它们的注意力权重”做加权合并 => 1 个补偿 token（K/V 各一个向量）
    - 最终顺序：[sink] + [comp] + [window]
    - 默认把 comp token 算入 max_capacity_prompt 预算：sink_size = max_capacity_prompt - window_size - 1
    """
    def __init__(self, window_size=64, max_capacity_prompt=256+64, kernel_size=5, pooling='avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.kernel_size = kernel_size   # 保留字段以兼容 config
        self.pooling = pooling           # 保留字段以兼容 config

    def reset(self, window_size=64, max_capacity_prompt=256+64, kernel_size=5, pooling='avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.kernel_size = kernel_size
        self.pooling = pooling

    @staticmethod
    def _apply_causal_mask_within_window(attn_weights, window_size):
        """
        仅对最后 window_size x window_size 的子块加因果 mask（与 SnapKV 思路一致）：
        attn_weights: (bsz, heads, window_size, q_len)
        """
        device = attn_weights.device
        dtype = attn_weights.dtype
        # (w, w) 上三角为 -inf（禁止看未来）
        mask = torch.full((window_size, window_size), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(window_size, device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(window_size, 1), 0)
        attn_weights[:, :, -window_size:, -window_size:] += mask[None, None, :, :]
        return attn_weights

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        # 形状一致性
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        # 缓存未满：不处理
        if q_len < self.max_capacity_prompt:
            return key_states, value_states

        # 预算内保留 1 个补偿 token
        comp_tokens = 1
        sink_size = self.window_size
        window_size = self.max_capacity_prompt - self.window_size - comp_tokens

        # 切片边界：
        # sink: [0, sink_size)
        # middle(drop): [sink_size, q_len - w)
        # window: [q_len - w, q_len)
        mid_start = sink_size
        mid_end = q_len - window_size
        middle_len = max(0, mid_end - mid_start)

        k_sink = key_states[:, :, :sink_size, :]
        v_sink = value_states[:, :, :sink_size, :]
        k_win  = key_states[:, :, -window_size:, :]
        v_win  = value_states[:, :, -window_size:, :]

        # 如果没有 middle token（比如 q_len 恰好接近预算），就退化为纯 sink+window（且不额外引入 comp）
        if middle_len == 0:
            # 这时为了不浪费预算，把 sink 放回 “不含 comp 的版本”
            sink_size2 = self.max_capacity_prompt - self.window_size
            if sink_size2 <= 0:
                return k_win, v_win
            k_sink2 = key_states[:, :, :sink_size2, :]
            v_sink2 = value_states[:, :, :sink_size2, :]
            return torch.cat([k_sink2, k_win], dim=2), torch.cat([v_sink2, v_win], dim=2)

        # -------- 核心：middle token merging（注意力加权合并）--------
        # 计算最近 window 的 query 对所有 key 的注意力分数
        # attn_logits: (bsz, heads, w, q_len)
        attn_logits = torch.matmul(
            query_states[:, :, -self.window_size:, :],
            key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)

        # 仅对窗口内部做因果 mask（middle 都在窗口之前，不需要额外 mask）
        attn_logits = self._apply_causal_mask_within_window(attn_logits, self.window_size)

        # softmax 得到注意力权重
        attn_probs = nn.functional.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # 将 window 内所有 query 对每个 key 的注意力求和，得到每个 token 的“重要性”
        # token_scores: (bsz, heads, q_len)
        token_scores = attn_probs.sum(dim=-2)

        # 取 middle 区间的权重并归一化（每个 head 单独归一化）
        mid_scores = token_scores[:, :, mid_start:mid_end]  # (bsz, heads, middle_len)
        denom = mid_scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        mid_w = mid_scores / denom  # (bsz, heads, middle_len)

        # 加权合并 K/V： (bsz, heads, middle_len, dim) * (bsz, heads, middle_len, 1)
        k_mid = key_states[:, :, mid_start:mid_end, :]
        v_mid = value_states[:, :, mid_start:mid_end, :]
        mid_w_e = mid_w.unsqueeze(-1)  # (bsz, heads, middle_len, 1)

        k_comp = (k_mid * mid_w_e).sum(dim=2, keepdim=True)  # (bsz, heads, 1, dim)
        v_comp = (v_mid * mid_w_e).sum(dim=2, keepdim=True)  # (bsz, heads, 1, dim)

        # 拼接：[sink] + [comp] + [window]
        key_out = torch.cat([k_sink, k_comp, k_win], dim=2)
        val_out = torch.cat([v_sink, v_comp, v_win], dim=2)

        # 保险：确保长度 == max_capacity_prompt（理论上应当成立）
        if key_out.shape[2] > self.max_capacity_prompt:
            key_out = key_out[:, :, -self.max_capacity_prompt:, :]
            val_out = val_out[:, :, -self.max_capacity_prompt:, :]

        return key_out, val_out


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