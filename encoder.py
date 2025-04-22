import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self , table_size , embedding_size , d_K , d_V):
        super(Encoder, self).__init__()

        self.d_K = d_K
        # 词嵌入表
        self.embedding = nn.Embedding(table_size , embedding_size)
        # Q K V 权重矩阵
        self.WQ = nn.Parameter(torch.randn(embedding_size , d_K))
        self.WK = nn.Parameter(torch.randn(embedding_size , d_K))
        self.WV = nn.Parameter(torch.randn(embedding_size , d_V))

        # 层标准化
        self.layer_norm = nn.LayerNorm(embedding_size)

        # 全连接神经网络
        self.FFN = nn.Sequential(
            nn.Linear(embedding_size , 2048),
            nn.ReLU(),
            nn.Linear(2048 , embedding_size)
        )


    # 自注意力 返回结合注意力的 V 矩阵
    def attention(self , input_embedding):
        # 计算 Q K V
        Q = input_embedding @ self.WQ
        K = (input_embedding @ self.WK).t()
        V = input_embedding @ self.WV
        # 计算注意力分数
        attention_score = Q @ K

        # 归一化
        attention_score /= torch.sqrt(torch.tensor(self.d_K))

        attention_score = torch.softmax(attention_score, dim=1)
        # 计算结果
        res = attention_score @ V

        return res

    # 残差连接和层归一化
    def __add_norm(self ,prev_input, cur_input):
        # 残差连接
        residuals = prev_input + cur_input
        # 层归一化
        res = self.layer_norm(residuals)

        return res



    def forward(self, input: list):
        # 获取词向量
        input = torch.tensor(input)
        embedding = self.embedding(input)

        # 自注意力
        attention_v = self.attention(embedding)

        # add & norm
        before_FFN = self.__add_norm(embedding, attention_v)

        # 全连接神经网络
        after_FFN = self.FFN(before_FFN)

        # add & norm
        output = self.__add_norm(before_FFN, after_FFN)

        return output



encoder = Encoder(table_size=1000, embedding_size=100, d_K=64, d_V=100)
encoder([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])












