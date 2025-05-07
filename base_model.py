import torch.nn as nn
from Attention import Attention

class BaseModel(nn.Module):
    def __init__(self , table_size , embedding_size , d_K , d_V):
        super(BaseModel, self).__init__()

        # 词嵌入表
        self.embedding = nn.Embedding(table_size , embedding_size)

        # 注意力机制
        self.attention = Attention(embedding_size , d_K , d_V)

        # 层标准化
        self.layer_norm = nn.LayerNorm(d_V)

        # 全连接神经网络
        self.FFN = nn.Sequential(
            nn.Linear(d_V , 4 * d_V),
            nn.ReLU(),
            nn.Linear(4 * d_V , d_V)
        )

    # 残差连接和层归一化
    def add_norm(self ,pre_input, cur_input):
        # 残差连接
        residuals = pre_input + cur_input
        # 层归一化
        res = self.layer_norm(residuals)

        return res
