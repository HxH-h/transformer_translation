import torch.nn as nn
from .Attention import Attention

class Decoder(nn.Module):

    def __init__(self , d_model , d_K , d_V , num_heads , dropout=0.2):
        super(Decoder , self).__init__()
        self.attention = Attention(d_model , d_K , d_V , num_heads)
        self.cross_attention = Attention(d_model , d_K , d_V , num_heads)
        self.attention_norm = nn.LayerNorm(d_model)
        self.cross_norm = nn.LayerNorm(d_model)
        self.add_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.FFN = nn.Sequential(
            nn.Linear(d_model , d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4 , d_model)
        )


    def forward(self , d_input , attention_mask , e_input = None , cross_mask = None):
        # 多头注意力
        attention_out = self.attention(d_input , d_input , attention_mask)
        attention_out = self.attention_norm(self.drop(attention_out) + d_input)
        # 交叉注意力
        cross_attention = None
        if e_input is not None:
            cross_attention = self.cross_attention(attention_out , e_input , cross_mask)
            cross_attention = self.cross_norm(self.drop(cross_attention) + attention_out)
        # 前馈网络
        x = attention_out if cross_attention is None else cross_attention
        out = self.FFN(x)
        out = self.add_norm(self.drop(out) + x)

        return out



