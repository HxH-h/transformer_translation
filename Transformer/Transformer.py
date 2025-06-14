from .encoder import Encoder
from .decoder import Decoder
import torch.nn as nn
import torch
from .Position import PositionEncode


class Transformer(nn.Module):
    def __init__(self , encoder_table_size , decoder_table_size,
                 embedding_size , d_K , d_V , num_heads ,
                 padding_idx , device):
        super(Transformer , self).__init__()
        self.device = device
        self.padding_idx = padding_idx
        # 位置编码
        self.position_encode = PositionEncode(embedding_size , device)
        # 编码层词向量表
        self.encoder_embed = nn.Embedding(encoder_table_size , embedding_size , padding_idx=padding_idx , device=device)
        # 解码层词向量表
        self.decoder_embed = nn.Embedding(decoder_table_size , embedding_size , padding_idx=padding_idx , device=device)
        # 编码器
        self.encoder = Encoder(embedding_size , d_K , d_V , num_heads)
        # 解码器
        self.decoder = Decoder(embedding_size , d_K , d_V , num_heads)
        # 线性层
        self.linear = nn.Linear(embedding_size , decoder_table_size)

    # input : (B , T)
    # return : padding_mask , tri_mask
    def generate_mask(self , e_input , d_input):
        db , dt = d_input.size()
        # mask 屏蔽 padding
        # 注意力的形状是(num_heads , B , T , _)
        padding_mask = (e_input == self.padding_idx).unsqueeze(0).unsqueeze(2)
        padding_mask = padding_mask.to(device=self.device)
        # mask  屏蔽未来信息
        mask = torch.triu(torch.ones(db, dt, dt, dtype=torch.bool , device=self.device), diagonal=1)

        return padding_mask, mask

    # English 为token list ，begin 为开始符索引， end 为结束符的索引
    def generate(self , English , begin , end , max):
        Chinese = [begin]
        # Encoder
        English  = torch.tensor(English , dtype=torch.long , device=self.device).unsqueeze(0)

        e_p = self.position_encode(English)
        e_embed = self.encoder_embed(English)
        e_output = self.encoder(e_embed + e_p , None)

        # Decoder
        i = 0
        while i < max:
            # 处理输入
            t = torch.tensor(Chinese , dtype=torch.long , device=self.device).unsqueeze(0)
            d_input = self.decoder_embed(t)
            d_p = self.position_encode(t)
            # 经过模型
            d_output = self.decoder(d_input = d_input + d_p ,
                                    attention_mask = None ,
                                    e_input = e_output)
            res = self.linear(d_output).squeeze(0)
            # softmax预测
            logits = res[-1]
            predict = torch.softmax(logits , dim=-1).argmax(dim=-1).item()
            if predict == end:
                break
            Chinese.append(predict)
            i += 1

        return Chinese




    def forward(self , e_input , d_input):
        # 生成掩码
        padding_mask , mask = self.generate_mask(e_input , d_input)
        # 位置编码
        e_p = self.position_encode(e_input)
        d_p = self.position_encode(d_input)
        # (B , T) -> (B , T , E)
        e_embed = self.encoder_embed(e_input)
        d_embed = self.decoder_embed(d_input)

        #print("encoder -------------------------------------")
        encoder_output = self.encoder(e_embed + e_p , padding_mask)
        #print("decoder -------------------------------------")
        decoder_output = self.decoder(d_input = d_embed + d_p ,
                                      attention_mask = mask ,
                                      e_input = encoder_output ,
                                      cross_mask = padding_mask)

        output = self.linear(decoder_output)
        return output










