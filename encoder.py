import torch
import torch.nn as nn
from Attention import Attention

class Encoder(nn.Module):
    def __init__(self , d_model , d_K , d_V , num_heads):
        super(Encoder, self).__init__()
        self.attention = Attention(d_model , d_K , d_V , num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model , d_model)

    # (B , T , d_model)
    def forward(self, input , mask):
        # 多头注意力
        attention = self.attention(input , input , mask)
        attention = self.norm1(input + attention)

        output = self.linear(attention)
        output = self.norm2(attention + output)

        return output














