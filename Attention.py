import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self , embedding_size , d_K , d_V):
         super(Attention , self).__init__()
         self.d_K = d_K
         # Q K V 权重矩阵
         self.WQ = nn.Parameter(torch.randn(embedding_size, d_K))
         self.WK = nn.Parameter(torch.randn(embedding_size, d_K))
         self.WV = nn.Parameter(torch.randn(embedding_size, d_V))


    def forward(self , input_Q , input_KV = None , mask = None):
        if input_KV is None:
            input_KV = input_Q

        # 计算 Q K V
        # (B , T , d_K)
        Q = input_Q @ self.WQ
        K = input_KV @ self.WK
        # (B , T , d_V)
        V = input_KV @ self.WV



        # 计算注意力分数
        #  (B , T , T)
        attention_score = Q @ K.transpose(-1, -2)


        # 归一化
        attention_score /= torch.sqrt(torch.tensor(self.d_K))

        # 掩码
        if mask is not None:
            attention_score = attention_score.masked_fill(mask, -float('inf'))
        print(attention_score)
        # 转为权重
        attention_score = torch.softmax(attention_score, dim= -1)

        print(attention_score)
        # 获取融入注意力的 V 矩阵
        # (B , T , T) @ (B , T , d_V) = (B , T , d_V)
        res = attention_score @ V



        return V , res


