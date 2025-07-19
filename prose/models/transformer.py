import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Absolute positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # decreases freq of sin encoding
        pe[:, 0::2] = torch.sin(position * div_term) # even indices
        pe[:, 1::2] = torch.cos(position * div_term) # odd indices

        self.register_buffer('pe', pe) # avoid being returned in model.parameters() and updating by optimizer

    def forward(self, x):
        seq_len = x.size(1) # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:seq_len, :].unsqueeze(0)
        return self.dropout(x)

class TransformerMLM(nn.Module):
    def __init__(self, input_dim, emb_dim, num_heads, num_layers, 
                dim_feedforward, dropout, out_dim, max_len):
        super(TransformerMLM, self).__init__()

        self.input_proj = nn.Linear(input_dim, emb_dim)
        self.emb_dim=emb_dim
        self.pos_encoder = PositionalEncoding(emb_dim, dropout, max_len)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )

        self.fc_out = nn.Linear(emb_dim, out_dim)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def forward(self, x, padding_mask):
        x = self.input_proj(x) # [batch_size, max_len, emb_dim]   
        x = self.pos_encoder(x) # [batch_size, max_len, emb_dim]
        emb = self.encoder(x, src_key_padding_mask=~padding_mask.bool())
        output = self.fc_out(emb)
        return output, emb

    def get_embedding_dim(self):
        return self.emb_dim


# Rotary positional encoding 
class RotaryEmbedding:
    def __init__(self, dim, max_len=500,base=10000):
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [seq_len, dim//2]

        # Precompute sin and cos
        self.sin = freqs.sin()[None, :, :]  # [1, seq_len, dim//2]
        self.cos = freqs.cos()[None, :, :]  # [1, seq_len, dim//2]

    def get_emb(self, device, seq_len):
        return (
            self.sin[:, :seq_len, :].to(device),
            self.cos[:, :seq_len, :].to(device),
        )

class MultiheadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, max_len=2048):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model // nheads must be an even number"

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(self.head_dim, max_len)

    def apply_rotary_pos_emb(self, x, sin, cos):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

    def forward(self, x, padding_mask):
        B, T, C = x.shape
        qkv = self.qkv_proj(x).view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # Each: [B, T, num_heads, head_dim]
        q = q.transpose(1, 2)  # → [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # print('q', q.shape)
        
        sin, cos = self.rope.get_emb(x.device, T)
        sin = sin.unsqueeze(1)  
        # print('sin', sin.shape)
        cos = cos.unsqueeze(1)

        q = self.apply_rotary_pos_emb(q, sin, cos)
        k = self.apply_rotary_pos_emb(k, sin, cos)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # print('att score', attn_scores.shape)
        # print('mask', padding_mask.shape)
        # raise Exception
        if padding_mask is not None:
            # mask = (padding_mask == 0) # 1: unpadded, 0: padded
            attn_scores = attn_scores.masked_fill(padding_mask[:, None, None, :], float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v  # [B, num_heads, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerEncoderLayerWithRoPE(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, max_len=2048):
        super().__init__()
        self.self_attn = MultiheadAttentionWithRoPE(d_model, nhead, dropout, max_len)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_key_padding_mask):
        src2 = self.self_attn(src, src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerMLM_RoPE(nn.Module):
    def __init__(self, input_dim, emb_dim, num_heads, num_layers, 
                 dim_feedforward, dropout, out_dim, max_len):
        super(TransformerMLM_RoPE, self).__init__()
        self.input_proj = nn.Linear(input_dim, emb_dim)
        self.emb_dim = emb_dim

        self.encoder = nn.ModuleList([
            TransformerEncoderLayerWithRoPE(
                d_model=emb_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_len=max_len
            ) for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(emb_dim, out_dim)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, padding_mask=None):
        x = self.input_proj(x)
        for layer in self.encoder:
            x = layer(x, src_key_padding_mask=~padding_mask.bool())
        output = self.fc_out(x)
        return output, x

    def get_embedding_dim(self):
        return self.emb_dim