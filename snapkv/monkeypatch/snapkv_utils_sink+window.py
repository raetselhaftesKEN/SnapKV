
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
    """
    对比基线：sink + sliding window
    - 保留最前面的 window_size 个 token（sink）
    - 保留最后 max - window_size 个 token（recent window）
    - 丢弃中间所有 token
    """
    def __init__(self, window_size=64, max_capacity_prompt=256+64, kernel_size=5, pooling='avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size >= 0
        # kernel_size/pooling 仅为保持配置兼容（这里不会用到）
        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(self, window_size=64, max_capacity_prompt=256+64, kernel_size=5, pooling='avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size >= 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        # 与 SnapKV 相同的形状假设
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        # 缓存未满：不裁剪
        if q_len < self.max_capacity_prompt:
            return key_states, value_states

        # 计算 sink_size，并执行 sink+window 裁剪
        sink_size = self.window_size
        window_size = self.max_capacity_prompt - self.window_size

        # 保留最前 sink_size + 最后 window_size（时间顺序保持不变）
        k_sink = key_states[:, :, :sink_size, :]
        v_sink = value_states[:, :, :sink_size, :]
        k_win  = key_states[:, :, -window_size:, :]
        v_win  = value_states[:, :, -window_size:, :]

        key_states = torch.cat([k_sink, k_win], dim=2)
        value_states = torch.cat([v_sink, v_win], dim=2)
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