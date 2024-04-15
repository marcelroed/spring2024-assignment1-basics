import torch
import torch.nn as nn
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
        input_dtype = x.dtype
        x = x.to(torch.float32)
        mean_squared = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(mean_squared + self.eps)
        return self.weight * x.to(input_dtype)
    

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / sqrt(2.0)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.activation = gelu
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
    
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


class MHASelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_pdrop: float | None = None, device=None):
        super().__init__()
        self.attn_pdrop = attn_pdrop
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        d_k = d_model // num_heads
        d_v = d_k  # Not necessarily the case
        self.d_k = d_k
        self.d_v = d_v
        self.W_qkv = nn.Parameter(torch.empty(3, num_heads, d_k, d_model, device=device))
        self.W_o = nn.Linear(num_heads * d_v, d_model, device=device)
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
                for head in range(self.num_heads):
                    weight = d[f"{qkvn}_proj.weight"]
                    weight = rearrange(weight, "(heads d) dm -> heads d dm", heads=self.num_heads)
                    self.W_qkv.data[qkvi, ...] = weight
        
        self.W_o.weight.data[:] = d["output_proj.weight"]

    
    def forward(self, x):
        seq_len = x.shape[-2]
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

        qkv_heads = einsum(x, self.W_qkv, "... s m, qkv h d m -> ... qkv h s d")
        Q, K, V = qkv_heads[..., 0, :, :, :], qkv_heads[..., 1, :, :, :], qkv_heads[..., 2, :, :, :]
        attn_output = sdpa(Q, K, V, mask=mask, pdrop=self.attn_pdrop)
        # attn_output: (..., heads, seq_len, dv)
        concatenated = rearrange(attn_output, "... h s d -> ... s (h d)")
        
        out = einsum(concatenated, self.W_o.weight, "... s hd, d hd -> ... s d")
        
        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, attn_pdrop: float | None = None, residual_pdrop: float | None = None):
        super().__init__()
        self.attn = MHASelfAttention(d_model, num_heads, attn_pdrop)
        self.ln1 = RMSNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.ln2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(residual_pdrop)
    
    def set_weights_from_dict(self, d):
        self.attn.set_weights_from_dict(dict_subset(d, "attn"))
        self.ln1.set_weights_from_dict(dict_subset(d, "ln1"))
        self.ln2.set_weights_from_dict(dict_subset(d, "ln2"))
        self.ffn.set_weights_from_dict(dict_subset(d, "ffn"))
    
    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, *, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int, d_ff: int, attn_pdrop: float | None = None, residual_pdrop: float | None = None):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, attn_pdrop=attn_pdrop, residual_pdrop=residual_pdrop) for _ in range(num_layers)])
        self.position_embedding = nn.Parameter(torch.zeros(context_length, d_model))
        self.dropout = nn.Dropout(residual_pdrop)
        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        print(f"{self=}")

    def set_weights_from_dict(self, d):
        self.token_embedding.weight.data[:] = d["token_embeddings.weight"]
        self.position_embedding.data[:] = d["position_embeddings.weight"]
        for i, block in enumerate(self.blocks):
            block.set_weights_from_dict(dict_subset(d, f"layers.{i}"))
        self.ln_final.set_weights_from_dict(dict_subset(d, "ln_final"))
        self.lm_head.weight.data[:] = d["lm_head.weight"]
    
    def forward(self, x):
        x = self.dropout(self.token_embedding(x) + self.position_embedding[None, :x.shape[-1], :])
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        x = softmax(x, dim=-1)
        return x
        
