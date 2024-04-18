from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from einops import einsum, rearrange, pack

def dict_subset(d, module):
    out_d = {}
    for k, v in d.items():
        if k.startswith(f'{module}.'):
            out_d[k[len(module)+1:]] = v
    return out_d

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None):
        # Identical to T5LayerNorm
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps
    
    def set_weights_from_dict(self, d):
        self.weight.data[:] = d["weight"]
    
    def forward(self, x):
        mean_squared = x.pow(2).mean(dim=-1, keepdim=True)
        x = x / torch.sqrt(mean_squared + self.eps)
        return self.weight * x
    

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / sqrt(2.0)))


class RotaryEmbeddings(nn.Module):
    def __init__(self, *, max_pos_idx: int, dim: int, device=None):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.base = 6000.0
        self.max_pos_idx = max_pos_idx

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))

        t = torch.arange(self.max_pos_idx, device=device, dtype=torch.int64).type_as(inv_freq)

        freqs = torch.outer(t, inv_freq)
        
        emb = torch.cat((freqs, freqs), dim=-1)

        dtype = torch.get_default_dtype()
        # Both of shape (max_pos_idx, dim)
        self.register_buffer('cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer('sin_cached', emb.sin().to(dtype), persistent=False)
    
    @staticmethod
    def rotate_half(x):
        # paper implementation:
        # rearrange([-qk_vecs[:, :, 1::2], qk_vecs[:, :, ::2]], 'sc total halfhidden -> total (halfhidden sc)')

        # in practice we don't need the immediate proximity of the above rearrange
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_embedding_to_qk_(self, qkv) -> None:
        # qkv: (..., 3, head, sequence, hidden_per_head)
        # assert qkv.shape[-1] == self.dim * 2, f'Are you sure you meant to use a different dim than hidden/2? Change this code if so. Found {self.dim=} with {qkv.shape=}'
        # print(pos_in_seq.max(), pos_in_seq.min())
        cos_values = self.cos_cached[None, None, None, :qkv.shape[-2], :]
        sin_values = self.sin_cached[None, None, None, :qkv.shape[-2], :]

        qk_vecs = qkv[..., :2, :, :, :self.dim]

        # print(f'{cos_values.shape=}, {qk_vecs.shape=}')
        cos_side = qk_vecs * cos_values
        sin_side = self.rotate_half(qk_vecs) * sin_values
        qk_positional_embedded = cos_side + sin_side
        # print(f'{qkv.shape=}, {qk_positional_embedded.shape=}')
        qkv[..., :2, :, :, :self.dim] = qk_positional_embedded

class GatedMLP(nn.Module):
    def __init__(self, d_model, d_ff, device=None):
        super().__init__()

        self.gate_proj = nn.Linear(d_model, d_ff, bias=False, device=device)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False, device=device)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False, device=device)
        self.activation = nn.SiLU()
    
    def forward(self, x):
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        gated_up = up * gate
        down = self.down_proj(gated_up)
        return down

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0, activation: Literal['gelu', 'silu']='gelu', device=None):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False, device=device)
        if activation == 'gelu':
            self.activation = gelu
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False, device=device)
    
    def set_weights_from_dict(self, d):
        self.linear1.weight.data = d["w1.weight"]
        self.linear2.weight.data = d["w2.weight"]
    
    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))

def softmax(x, dim):
    # Numerically stable softmax
    x_max = x.max(dim=dim, keepdim=True).values
    x_adjusted = x - x_max
    x_exp = torch.exp(x_adjusted)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

def sdpa(Q, K, V, mask, pdrop):
    qk_prod = einsum(Q, K, "... qs d,... ks d -> ... qs ks")
    if mask is not None:
        qk_prod = qk_prod.masked_fill(mask, -float("inf"))
    attn_weights = softmax(qk_prod / sqrt(Q.shape[-1]), dim=-1)
    if pdrop is not None:
        attn_weights = nn.functional.dropout(attn_weights, pdrop)
    return einsum(attn_weights, V, "... qs ks,... ks d -> ... qs d")


class MHSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, context_length: int, attn_pdrop: float | None = None, use_flash: bool = False, use_rotary_embeddings: bool = False, device=None):
        super().__init__()
        self.attn_pdrop = attn_pdrop
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        d_k = d_model // num_heads
        d_v = d_k  # Not necessarily the case
        self.d_k = d_k
        self.d_v = d_v
        self.use_flash = use_flash
        self.W_qkv = nn.Parameter(torch.empty(3, num_heads, d_k, d_model, device=device))
        self.W_o = nn.Linear(num_heads * d_v, d_model, bias=False, device=device)
        if use_rotary_embeddings:
            self.rotary_embeddings = RotaryEmbeddings(max_pos_idx=context_length, dim=d_k // 2, device=device)
        self.reset_parameters()
    
    def reset_parameters(self):
        for qkv in range(self.W_qkv.shape[0]):
            for head in range(self.W_qkv.shape[1]):
                nn.init.kaiming_uniform_(self.W_qkv[qkv, head], a=sqrt(5))
    
    def set_weights_from_dict(self, d):
        if 'q_heads.0.weight' in d:
            for qkvi, qkvn in enumerate('qkv'):
                for head in range(self.num_heads):
                    self.W_qkv.data[qkvi, head] = d[f"{qkvn}_heads.{head}.weight"]
        else:
            for qkvi, qkvn in enumerate('qkv'):
                weight = d[f"{qkvn}_proj.weight"]
                weight = rearrange(weight, "(heads d) dm -> heads d dm", heads=self.num_heads)
                self.W_qkv.data[qkvi, ...] = weight
        
        self.W_o.weight.data[:] = d["output_proj.weight"]

    
    def forward(self, x):
        seq_len = x.shape[-2]

        qkv_heads = einsum(x, self.W_qkv, "... s m, qkv h d m -> ... qkv h s d")
        if hasattr(self, 'rotary_embeddings'):
            self.rotary_embeddings.apply_rotary_embedding_to_qk_(qkv_heads)
        Q, K, V = qkv_heads[..., 0, :, :, :], qkv_heads[..., 1, :, :, :], qkv_heads[..., 2, :, :, :]
        if self.use_flash:
            attn_output = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.attn_pdrop or 0, is_causal=True)
        else:
            mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
            attn_output = sdpa(Q, K, V, mask=mask, pdrop=self.attn_pdrop)
        # attn_output: (..., heads, seq_len, dv)
        concatenated = rearrange(attn_output, "... head token d -> ... token (head d)")
        
        out = einsum(concatenated, self.W_o.weight, "... token head, d head -> ... token d")
        
        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, context_length: int = None, use_rotary_embeddings: bool = False, use_gated_mlp: bool = False,  attn_pdrop: float | None = None, activation: Literal['gelu', 'silu'] = 'gelu', use_flash: bool = False, residual_pdrop: float | None = None, parallel_layers: bool = False, post_norm: bool = False, device=None):
        super().__init__()
        self.attn = MHSelfAttention(d_model=d_model, num_heads=num_heads, attn_pdrop=attn_pdrop, use_flash=use_flash, context_length=context_length, use_rotary_embeddings=use_rotary_embeddings, device=device)
        self.ln1 = RMSNorm(d_model, device=device)
        if use_gated_mlp:
            self.ffn = GatedMLP(d_model, d_ff, device=device)
        else:
            self.ffn = PositionwiseFeedForward(d_model, d_ff, activation=activation, device=device)
        self.ln2 = RMSNorm(d_model, device=device)
        self.dropout = nn.Dropout(residual_pdrop or 0.0)
        self.parallel_layers = parallel_layers
        self.post_norm = post_norm
    
    def set_weights_from_dict(self, d):
        self.attn.set_weights_from_dict(dict_subset(d, "attn"))
        self.ln1.set_weights_from_dict(dict_subset(d, "ln1"))
        self.ln2.set_weights_from_dict(dict_subset(d, "ln2"))
        self.ffn.set_weights_from_dict(dict_subset(d, "ffn"))
    
    def forward(self, x):
        if self.parallel_layers:
            x = x + self.dropout(self.attn(self.ln1(x))) + self.dropout(self.ffn(self.ln2(x)))
        elif self.post_norm:
            x = self.ln1(x + self.dropout(self.attn(x)))
            x = self.ln2(x + self.dropout(self.ffn(x)))
        else:
            x = x + self.dropout(self.attn(self.ln1(x)))
            x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, *, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, activation: Literal['gelu', 'silu'] = 'gelu', tie_embeddings: bool = False, use_gated_mlp: bool = False, use_rotary_embeddings: bool = False, attn_pdrop: float | None = None, residual_pdrop: float | None = None, parallel_layers: bool = False, post_norm: bool = False, device=None, use_flash: bool = False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, device=device)
        if tie_embeddings:
            nn.init.kaiming_uniform_(self.token_embedding.weight, a=sqrt(5))

        self.blocks = nn.Sequential(*[TransformerBlock(d_model=d_model, num_heads=num_heads, activation=activation, d_ff=d_ff, attn_pdrop=attn_pdrop, residual_pdrop=residual_pdrop,
                                                       parallel_layers=parallel_layers, post_norm=post_norm, device=device, use_flash=use_flash, use_rotary_embeddings=use_rotary_embeddings,
                                                       context_length=context_length, use_gated_mlp=use_gated_mlp) for _ in range(num_layers)])
        self.tie_embeddings = tie_embeddings
        if not use_rotary_embeddings:
            self.position_embedding = nn.Parameter(torch.zeros(context_length, d_model, device=device))
        self.dropout = nn.Dropout(residual_pdrop or 0.0)
        self.ln_final = RMSNorm(d_model, device=device)
        if not self.tie_embeddings:
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False, device=device)
        print(f"{self=}")

    def set_weights_from_dict(self, d):
        self.token_embedding.weight.data[:] = d["token_embeddings.weight"]
        self.position_embedding.data[:] = d["position_embeddings.weight"]
        for i, block in enumerate(self.blocks.children()):
            block.set_weights_from_dict(dict_subset(d, f"layers.{i}"))
        assert f'layers.{i + 1}' not in d, "Extra weights in state dict"
        self.ln_final.set_weights_from_dict(dict_subset(d, "ln_final"))
        self.lm_head.weight.data[:] = d["lm_head.weight"]
    
    def forward(self, x):
        if hasattr(self, 'position_embedding'):
            x = self.dropout(self.token_embedding(x) + self.position_embedding[None, :x.shape[-1], :])
        else:
            x = self.dropout(self.token_embedding(x))
        x = self.blocks(x)
        x = self.ln_final(x)
        if self.tie_embeddings:
            x = F.linear(x, self.token_embedding.weight)
        else:
            x = self.lm_head(x)
        # x = softmax(x, dim=-1)
        return x
        
