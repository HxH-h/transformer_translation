from base_model import BaseModel
import torch
from Attention import Attention

class Decoder(BaseModel):

    def __init__(self , table_size , embedding_size , d_K , d_V):
        super(Decoder , self).__init__(table_size , embedding_size , d_K , d_V)

        self.cross_attention = Attention(embedding_size , d_K , d_V)

    def forward(self , de_input , en_input):
        # 获取词向量
        de_input = torch.tensor(de_input)
        de_embedding_input = self.embedding(de_input)

        # 掩码 注意力
        attention_size = de_input.size(-1)

        mask = torch.triu(torch.ones(attention_size , attention_size, dtype=torch.bool) , diagonal = 1)

        # 自注意力 返回 原始的 V 和根据注意力得到的 V
        V, attention_v = self.attention(de_embedding_input , mask = mask)

        # add & norm
        de_before_cross = self.add_norm(V, attention_v)

        # 交叉注意力
        after_cross , _ =self.cross_attention(input_Q = de_before_cross ,
                                                input_KV = en_input)

        before_FFN = self.add_norm(de_before_cross , after_cross)

        # 全连接神经网络
        after_FFN = self.FFN(before_FFN)

        # add & norm
        output = self.add_norm(before_FFN, after_FFN)



