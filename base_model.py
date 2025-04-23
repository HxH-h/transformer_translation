import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self , table_size , embedding_size , d_K , d_V):
        super(BaseModel, self).__init__()

        self.d_K = d_K
        # 词嵌入表
        self.embedding = nn.Embedding(table_size , embedding_size)
        # Q K V 权重矩阵
        self.WQ = nn.Parameter(torch.randn(embedding_size , d_K))
        self.WK = nn.Parameter(torch.randn(embedding_size , d_K))
        self.WV = nn.Parameter(torch.randn(embedding_size , d_V))

        # 层标准化
        self.layer_norm = nn.LayerNorm(d_V)

        # 全连接神经网络
        self.FFN = nn.Sequential(
            nn.Linear(d_V , 4 * d_V),
            nn.ReLU(),
            nn.Linear(4 * d_V , d_V)
        )

    # 自注意力 返回结合注意力的 V 矩阵
    def attention(self , input_embedding , mask = None):
        # 计算 Q K V
        Q = input_embedding @ self.WQ
        K = (input_embedding @ self.WK).t()
        V = input_embedding @ self.WV
        # 计算注意力分数
        attention_score = Q @ K

        # 归一化
        attention_score /= torch.sqrt(torch.tensor(self.d_K))

        # 掩码
        if mask is not None:
            attention_score = attention_score.masked_fill(mask, -float('inf'))

        attention_score = torch.softmax(attention_score, dim=1)

        # 计算结果
        res = attention_score @ V

        return V , res

    # 残差连接和层归一化
    def add_norm(self ,pre_input, cur_input):
        # 残差连接
        residuals = pre_input + cur_input
        # 层归一化
        res = self.layer_norm(residuals)

        return res
