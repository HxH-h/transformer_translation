import torch.nn as nn
import torch
import math
class PositionEncode(nn.Module):
    def __init__(self, d_model, device , max_len=5000):
        super(PositionEncode, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model , device=device)
        position = torch.arange(0, max_len).unsqueeze(1).to(device=device)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model)).to(device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        B , T = x.shape
        return self.pe[:, :T, :]


