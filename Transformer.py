from encoder import Encoder
from decoder import Decoder
import torch.nn as nn
import torch


class Transformer(nn.Module):
    def __init__(self , table_size , embedding_size , d_K , d_V , num_heads):
        super(Transformer , self).__init__()
        # 词向量表
        self.embed = nn.Embedding(table_size , embedding_size)
        # 编码器
        self.encoder = Encoder(embedding_size , d_K , d_V , num_heads)
        # 解码器
        self.decoder = Decoder(embedding_size , d_K , d_V , num_heads)
        # 线性层
        self.linear = nn.Linear(embedding_size , table_size)

    # input : (B , T)
    # return : padding_mask , tri_mask
    def generate_mask(self , input):
        b , t = input.size()
        # mask 屏蔽 padding
        lengths = (input != 0).int().argmax(dim=-1)

        m = torch.arange(t).expand(len(lengths), t) < lengths.unsqueeze(1)
        padding_mask = m.unsqueeze(1).expand(-1, t, -1)

        # mask  屏蔽未来信息
        mask = torch.triu(torch.ones(b, t, t, dtype=torch.bool), diagonal=1)

        return padding_mask, mask


    def forward(self , e_input , d_input):
        # 生成掩码
        padding_mask , mask = self.generate_mask(d_input)
        # (B , T) -> (B , T , E)
        e_embed = self.embed(e_input)
        d_embed = self.embed(d_input)

        encoder_output = self.encoder(e_embed , padding_mask)

        decoder_output = self.decoder(d_input = d_embed ,
                                      attention_mask = mask ,
                                      e_input = encoder_output ,
                                      cross_mask = padding_mask)

        output = self.linear(decoder_output)
        return output










