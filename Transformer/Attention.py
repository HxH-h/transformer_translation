import torch
import torch.nn as nn
# 多头注意力
class Attention(nn.Module):
    def __init__(self , d_model , d_K , d_V , num_heads):
         super(Attention , self).__init__()
         self.d_K = d_K
         self.num_heads = num_heads
         # Q K V 权重矩阵
         self.WQ = nn.Linear(d_model, d_K * num_heads)
         self.WK = nn.Linear(d_model, d_K * num_heads)
         self.WV = nn.Linear(d_model, d_V * num_heads)
         # 输出权重矩阵
         self.W_O = nn.Linear(d_V * num_heads, d_model)

    def single_attention(self , Q , K , V , mask = None):
        # (B , T , d_K) @ (B , d_K , T) -> (B , T , T)
        attention_score = Q @ K.transpose(-1, -2)

        #print("注意力得分",attention_score.shape)
        attention_score /= torch.sqrt(torch.tensor(self.d_K))
        # 掩码
        if mask is not None:
            attention_score = attention_score.masked_fill(mask, -float('inf'))
        #print("注意力得分",attention_score)
        # 转为权重
        attention_score = torch.softmax(attention_score, dim= -1)

        # (B , T , T) @ (B , T , d_V) = (B , T , d_V)
        return attention_score @ V

    # 分割多头注意力
    def split_heads(self , input , num_heads):
        b , t , d_model = input.size()
        # (B , T , d_ * num_heads) -> (B , T , num_heads , d_)
        w = input.view(b, t, num_heads, -1)

        # (B , T , num_heads , d_) -> (num_heads , B , T , d_)
        # 深拷贝 w
        w = w.permute(2, 0, 1, 3).contiguous()

        return w
    def combine_heads(self , input):
        n , b , t , d_V = input.size()

        # (num_heads , B , T , d_V) -> (B , T , num_heads , d_V)
        w = input.permute(1, 2, 0, 3).contiguous()
        #print("注意力头移到后面",w.shape)
        # (B , T , num_heads , d_V) -> (B , T , d_V * num_heads)
        w = w.view(b, t, -1)
        #print("合并注意力头",w.shape)

        return w


    # (B , T , d_model)
    def forward(self , input_Q , input_KV = None , mask = None):
        if input_KV is None:
            input_KV = input_Q

        # 计算 Q K V
        # (B , T , d_K * num_heads)
        Q = self.WQ(input_Q)
        K = self.WK(input_KV)
        # (B , T , d_V * num_heads)
        V = self.WV(input_KV)

        # 拆分注意力 转为单头注意力并行计算
        # (B , T , d_ * num_heads) -> (num_heads , B , T , d_)
        Q = self.split_heads(Q, self.num_heads)
        K = self.split_heads(K, self.num_heads)
        V = self.split_heads(V, self.num_heads)

        # 单头注意力
        # (num_heads , B , T , d_V)
        attention_score = self.single_attention(Q, K, V, mask)
        #print("注意力结果",attention_score.shape)

        # 合并多头注意力
        # (num_heads , B , T , d_V) -> (B , T , d_V * num_heads)
        combine = self.combine_heads(attention_score)
        #print("合并的结果",combine.shape)
        #  (B , T , d_V * num_heads) -> (B , T , d_model)
        output = self.W_O(combine)
        #print("转为d_model大小",output.shape)

        return output



