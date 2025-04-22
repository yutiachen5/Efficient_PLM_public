import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

