import torch
from base_model import BaseModel

class Encoder(BaseModel):
    def __init__(self , table_size , embedding_size , d_K , d_V):
        super(Encoder, self).__init__(table_size, embedding_size, d_K, d_V)

    def forward(self, input: list):
        # 获取词向量
        input = torch.tensor(input)
        embedding = self.embedding(input)

        # 自注意力 返回 原始的 V 和根据注意力得到的 V
        V , attention_v = self.attention(embedding)

        # add & norm
        before_FFN = self.add_norm(V, attention_v)

        # 全连接神经网络
        after_FFN = self.FFN(before_FFN)

        # add & norm
        output = self.add_norm(before_FFN, after_FFN)
        print(output.shape)
        return output



encoder = Encoder(table_size=1000, embedding_size=100, d_K=64, d_V=64)
encoder([1, 2, 3, 4, 5])












