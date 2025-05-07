import torch
from base_model import BaseModel

class Encoder(BaseModel):
    def __init__(self , table_size , embedding_size , d_K , d_V):
        super(Encoder, self).__init__(table_size, embedding_size, d_K, d_V)

    def forward(self, input):
        # 获取词向量(B , T)
        input = torch.tensor(input)
        # mask 屏蔽 padding
        lengths = (input != 0).int().argmax(dim=-1)
        max_len = input.size(-1)
        m = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        mask = m.unsqueeze(1).expand(-1, input.size(-1), -1)

        # 词嵌入 (B , T , E)
        embedding_input = self.embedding(input)

        # 自注意力 返回 原始的 V 和根据注意力得到的 V
        V , attention_v = self.attention(embedding_input , mask = mask)

        # add & norm
        before_FFN = self.add_norm(V, attention_v)


        # 全连接神经网络
        after_FFN = self.FFN(before_FFN)

        # add & norm
        output = self.add_norm(before_FFN, after_FFN)


        return output , mask














