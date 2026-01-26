
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
from typing import Tuple

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
    """SnapKV cache compressor + (optional) dropped-token compensation.

    Compensation implements the feature-map formulation we derived:

        O_hat(q) = (N_C + N_D_hat) / (Z_C + Z_D_hat)

    where (N_D_hat, Z_D_hat) are recovered from two summaries S, M maintained
    over dropped tokens using a positive random feature map phi for the softmax
    kernel.
    """

    def __init__(
        self,
        window_size: int = 64,
        max_capacity_prompt: int = 256 + 64,
        kernel_size: int = 5,
        pooling: str = 'avgpool',
        # --- compensation configs ---
        comp_enabled: bool = False,
        comp_rff_dim: int = 128,
        comp_seed: int = 42,
    ):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

        # compensation
        self.comp_enabled = bool(comp_enabled)
        self.comp_rff_dim = int(comp_rff_dim)
        self.comp_seed = int(comp_seed)
        self._comp_inited = False
        self._W = None          # (num_heads, r, head_dim)
        self._S = None          # (bsz, num_heads, r)
        self._M = None          # (bsz, num_heads, r, head_dim)
        self._rff_scale = None  # scalar = d^{-1/4}, set when init

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

        # reset compensation state (keep config)
        self._comp_inited = False
        self._W = None
        self._S = None
        self._M = None
        self._rff_scale = None

    # ----------------------------
    # Compensation: positive random features for softmax kernel
    # ----------------------------
    def _maybe_init_comp(self, bsz: int, num_heads: int, head_dim: int, device, dtype):
        """Initialize per-layer per-head RFF parameters and summaries."""
        if (not self.comp_enabled) or self._comp_inited:
            return

        if self.comp_rff_dim <= 0:
            raise ValueError(f"comp_rff_dim must be > 0, got {self.comp_rff_dim}")

        # Scale so that (q_scaled · k_scaled) = (q · k) / sqrt(d)
        # to match attention logits qk^T / sqrt(d).
        self._rff_scale = float(head_dim) ** (-0.25)

        # Per-head Gaussian random projection matrices (fixed after init).
        g = torch.Generator(device='cpu')
        g.manual_seed(self.comp_seed)
        W = torch.randn(num_heads, self.comp_rff_dim, head_dim, generator=g, dtype=torch.float32)
        self._W = W.to(device=device)

        # Per-batch summaries. We keep float32 for stability; cast at use-site if needed.
        self._S = torch.zeros(bsz, num_heads, self.comp_rff_dim, device=device, dtype=torch.float32)
        self._M = torch.zeros(bsz, num_heads, self.comp_rff_dim, head_dim, device=device, dtype=torch.float32)

        self._comp_inited = True

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """Compute positive random feature map phi(x).

        x: (bsz, heads, L, head_dim) or (bsz, heads, head_dim)
        returns: (..., r)
        """
        # Ensure initialized
        assert self._W is not None and self._rff_scale is not None
        r = self.comp_rff_dim

        if x.dim() == 3:
            x_in = x.unsqueeze(2)  # (bsz, heads, 1, d)
            squeeze_L = True
        elif x.dim() == 4:
            x_in = x
            squeeze_L = False
        else:
            raise ValueError(f"phi expects x dim 3 or 4, got {x.dim()}")

        x_scaled = x_in * self._rff_scale
        # proj: (bsz, heads, L, r)
        proj = torch.einsum('bhld,hrd->bhlr', x_scaled.to(torch.float32), self._W)
        # norm: (bsz, heads, L, 1)
        norm = 0.5 * (x_scaled.to(torch.float32) ** 2).sum(dim=-1, keepdim=True)
        phi = torch.exp(proj - norm) / math.sqrt(r)

        if squeeze_L:
            return phi.squeeze(2)
        return phi

    def comp_update_dropped(self, dropped_k: torch.Tensor, dropped_v: torch.Tensor):
        """Accumulate dropped tokens into (S, M).

        dropped_k/v: (bsz, heads, L_drop, head_dim)
        """
        if (not self.comp_enabled) or (dropped_k is None) or (dropped_k.numel() == 0):
            return
        assert self._comp_inited
        # phi_k: (bsz, heads, L_drop, r)
        phi_k = self._phi(dropped_k)
        # S += sum phi(k)
        self._S = self._S + phi_k.sum(dim=2)
        # M += sum phi(k) v^T
        self._M = self._M + torch.einsum('bhlr,bhld->bhrd', phi_k, dropped_v.to(torch.float32))

    def comp_terms(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute raw (Z_D_hat, N_D_hat) for current query.

        query: (bsz, heads, q_len, head_dim) or (bsz, heads, head_dim)
        returns:
          Z_D_raw: (bsz, heads, q_len) or (bsz, heads)
          N_D_raw: (bsz, heads, q_len, head_dim) or (bsz, heads, head_dim)
        """
        if (not self.comp_enabled) or (not self._comp_inited):
            return None, None
        u = self._phi(query)  # (bsz, heads, q_len, r) or (bsz, heads, r)
        if u.dim() == 3:
            # (bsz, heads, r)
            Z = (u * self._S).sum(dim=-1)  # (bsz, heads)
            N = torch.einsum('bhr,bhrd->bhd', u, self._M)  # (bsz, heads, d)
            return Z, N
        else:
            # (bsz, heads, q_len, r)
            Z = torch.einsum('bhlr,bhr->bhl', u, self._S)  # (bsz, heads, q_len)
            N = torch.einsum('bhlr,bhrd->bhld', u, self._M)  # (bsz, heads, q_len, d)
            return Z, N

    '''
    核心函数：压缩kv缓存
    整个kv缓存的操作都在这里面
    你要加入token补偿这些的，就也在这里面加，应该核心向量是attn_weights
    '''
    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        # init compensation state (only if enabled)
        self._maybe_init_comp(bsz=bsz, num_heads=num_heads, head_dim=head_dim, device=key_states.device, dtype=key_states.dtype)
        if q_len < self.max_capacity_prompt:  # 缓存没满，不压缩
            return key_states, value_states
        else:
            # attn_weights：只看window_size个token
            # 下面这一行就在计算观察窗口对全部token的注意力
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)

            # 然后计算观察窗口内部的注意力
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            # 两个注意力拼起来
            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask
            # softmax正则化
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # 对旧的token计算观察窗口的注意力之和（用于舍弃）
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)

            # 下面是做一个1D池化平滑，防止单点噪声
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')

            # 从旧token选取topK位置
            keep_len = self.max_capacity_prompt - self.window_size
            old_len = key_states.shape[-2] - self.window_size

            # Use argsort to obtain an exact partition into {kept, dropped}
            # (avoids potential overlap with topk/bottomk under ties).
            sorted_idx = attn_cache.argsort(dim=-1, descending=True)  # (bsz, heads, old_len)
            top_idx = sorted_idx[:, :, :keep_len]  # (bsz, heads, keep_len)
            drop_idx = sorted_idx[:, :, keep_len:]  # (bsz, heads, drop_len)
            indices = top_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            # 把kv缓存把选中的topK token聚集到一起（k、v向量各做一次）
            old_k = key_states[:, :, :-self.window_size, :]
            old_v = value_states[:, :, :-self.window_size, :]
            k_past_compress = old_k.gather(dim=2, index=indices)
            v_past_compress = old_v.gather(dim=2, index=indices)

            # --- compensation update: summarize dropped tokens (old part excluding the window) ---
            drop_len = old_len - keep_len
            if self.comp_enabled and drop_len > 0:
                drop_idx_exp = drop_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                dropped_k = old_k.gather(dim=2, index=drop_idx_exp)
                dropped_v = old_v.gather(dim=2, index=drop_idx_exp)
                self.comp_update_dropped(dropped_k, dropped_v)

            # 然后拼接topK和观察窗口的kv缓存
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
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
        # --- compensation defaults (can be overridden by your compress_args/config) ---
        if not hasattr(self.config, 'comp_enabled'):
            self.config.comp_enabled = True
        if not hasattr(self.config, 'comp_rff_dim'):
            self.config.comp_rff_dim = 128
        if not hasattr(self.config, 'comp_seed'):
            self.config.comp_seed = 42
    self.kv_cluster = SnapKVCluster(
        window_size = self.config.window_size,
        max_capacity_prompt = self.config.max_capacity_prompt,
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling,
        comp_enabled = self.config.comp_enabled,
        comp_rff_dim = self.config.comp_rff_dim,
        comp_seed = self.config.comp_seed,
        )