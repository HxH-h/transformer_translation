from base_model import BaseModel
import torch


class Decoder(BaseModel):

    def __init__(self , table_size , embedding_size , d_K , d_V):
        super(Decoder , self).__init__(table_size , embedding_size , d_K , d_V)

    def forword(self , input):
        # 获取词向量
        input = torch.tensor(input)
        embedding = self.embedding(input)

        # 掩码 注意力
        attention_size = input.size(-1)

        mask = torch.triu(torch.ones(attention_size , attention_size, dtype=torch.bool) , diagonal = 1)

        # 自注意力 返回 原始的 V 和根据注意力得到的 V
        V, attention_v = self.attention(embedding , mask)

        # add & norm
        before_cross = self.add_norm(V, attention_v)



decoder = Decoder(100 , 100 , 100 , 100)

decoder.forword([1 , 2 , 3 , 4 , 5])

