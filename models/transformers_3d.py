import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Transformer3D(nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 6, dim_feedforward: int = 2048):
        super().__init__()
        self.d_model = d_model
        
        self.input_proj = nn.Linear(3, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.positional_encoding = PositionalEncoding3D(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = x.transpose(0, 1)
        
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        
        x = torch.max(x, 1)[0]
        return x

class PositionalEncoding3D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]

class SetTransformer(nn.Module):
    def __init__(self, dim_input: int = 3, num_outputs: int = 1, dim_output: int = 128,
                 num_inds: int = 32, dim_hidden: int = 128, num_heads: int = 4, num_blocks: int = 2):
        super().__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, num_blocks),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, num_blocks),
        )
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, num_blocks),
            SAB(dim_hidden, dim_hidden, num_heads, num_blocks),
            nn.Linear(dim_hidden, dim_output)
        )
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(X))

class MAB(nn.Module):
    def __init__(self, dim_Q: int, dim_K: int, dim_V: int, num_heads: int, ln: bool = False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        
        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, num_blocks: int, ln: bool = False):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, num_inds: int, num_blocks: int, ln: bool = False):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_seeds: int, num_blocks: int, ln: bool = False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)