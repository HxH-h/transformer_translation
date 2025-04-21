import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self , table_size , embedding_size , d_QK , d_V):
        super(Encoder, self).__init__()
        # 词嵌入表
        self.embedding = nn.Embedding(table_size , embedding_size)
        # Q K V 权重矩阵
        self.WQ = nn.Parameter(torch.randn(embedding_size , d_QK))
        self.WK = nn.Parameter(torch.randn(embedding_size , d_QK))
        self.WV = nn.Parameter(torch.randn(embedding_size , d_V))

    def normalization(self , input):
        mean = torch.mean(input, dim=1, keepdim=True)
        std = torch.std(input, dim=1, keepdim=True)
        return (input - mean) / std

    # 自注意力 返回结合注意力的 V 矩阵
    def attention(self , input_embedding):
        # 计算 Q K V
        Q = input_embedding @ self.WQ
        K = (input_embedding @ self.WK).t()
        V = input_embedding @ self.WV
        print("Q", Q.shape)
        print("K", K.shape)
        print("V", V.shape)
        # 计算注意力分数
        attention_score = Q @ K
        print("attention_score", attention_score.shape)

        # 归一化
        attention_score = self.normalization(attention_score)

        attention_score = torch.softmax(attention_score, dim=1)
        # 计算结果
        res = attention_score @ V

        return res

    def forward(self, input: list):
        # 获取词向量
        input = torch.tensor(input)
        embedding = self.embedding(input)

        # 自注意力
        attention_v = self.attention(embedding)

        # 残差连接
        residuals = embedding + attention_v


        return attention_v















