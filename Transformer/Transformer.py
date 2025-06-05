from .encoder import Encoder
from .decoder import Decoder
import torch.nn as nn
import torch
from .Position import PositionEncode

class Transformer(nn.Module):
    def __init__(self , encoder_table_size , decoder_table_size,
                 embedding_size , d_K , d_V , num_heads ,
                 encoder_padding_idx , decoder_padding_idx):
        super(Transformer , self).__init__()
        self.padding_idx = encoder_padding_idx
        # 位置编码
        self.position_encode = PositionEncode(embedding_size)
        # 编码层词向量表
        self.encoder_embed = nn.Embedding(encoder_table_size , embedding_size , padding_idx=encoder_padding_idx)
        # 解码层词向量表
        self.decoder_embed = nn.Embedding(decoder_table_size , embedding_size , padding_idx=decoder_padding_idx)
        # 编码器
        self.encoder = Encoder(embedding_size , d_K , d_V , num_heads)
        # 解码器
        self.decoder = Decoder(embedding_size , d_K , d_V , num_heads)
        # 线性层
        self.linear = nn.Linear(embedding_size , decoder_table_size)

    # input : (B , T)
    # return : padding_mask , tri_mask
    def generate_mask(self , e_input , d_input):
        db , dt = d_input.size()
        # mask 屏蔽 padding
        # 注意力的形状是(num_heads , B , T , _)
        padding_mask = (e_input == self.padding_idx).unsqueeze(0).unsqueeze(2)
        # mask  屏蔽未来信息
        mask = torch.triu(torch.ones(db, dt, dt, dtype=torch.bool), diagonal=1)

        return padding_mask, mask


    def forward(self , e_input , d_input):
        # 生成掩码
        padding_mask , mask = self.generate_mask(e_input , d_input)
        # 位置编码
        e_p = self.position_encode(e_input)
        d_p = self.position_encode(d_input)
        # (B , T) -> (B , T , E)
        e_embed = self.encoder_embed(e_input)
        d_embed = self.decoder_embed(d_input)

        #print("encoder -------------------------------------")
        encoder_output = self.encoder(e_embed + e_p , padding_mask)
        #print("decoder -------------------------------------")
        decoder_output = self.decoder(d_input = d_embed + d_p ,
                                      attention_mask = mask ,
                                      e_input = encoder_output ,
                                      cross_mask = padding_mask)

        output = self.linear(decoder_output)
        return output










