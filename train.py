import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from Transformer.Transformer import Transformer
from tokenizer import Tokenizer
import utils
import pandas as pd
#%% 数据处理
# 分词器
tokenizer = Tokenizer()
en , zh = tokenizer.add_special_tokens(['<begin>','<end>','<pad>'])
print("分词器加载完成")

# 处理数据
data = utils.get_data('./translation2019zh/t.json')

tokens = pd.DataFrame()
tokens['en'] = data['en'].apply(lambda x: tokenizer.encode(x))
tokens['zh'] = data['zh'].apply(lambda x: tokenizer.encode_chinese('<begin>' + x))
print(data.head())
print(tokens.head())
print("完成分词")

# 创建 dataset 和 dataloader

class Trans_dataset(Dataset):
    def __init__(self , data , end):
        self.data = data
        self.end = end
    def __getitem__(self , index):
        return self.data['en'][index] , self.data['zh'][index] , self.data['zh'][index][1:] + [self.end]
    def __len__(self):
        return len(self.data)

def custom_collate(batch):
    # batch 中的每个元素是一个样本
    # 提取输入数据并进行填充

    x , y , z = zip(*batch)

    x = pad_sequence([torch.tensor(i) for i in x], batch_first=True , padding_value=en['<pad>'])
    y = pad_sequence([torch.tensor(i) for i in y], batch_first=True , padding_value=zh['<pad>'])
    z = pad_sequence([torch.tensor(i) for i in z], batch_first=True , padding_value=zh['<pad>'])
    # for data in zip(*batch):
    #     # 返回填充后的 batch 数据
    #     yield pad_sequence([torch.tensor(i) for i in data], batch_first=True , padding_value=zh['<pad>'])
    return x , y , z

dataset  = Trans_dataset(tokens , zh['<end>'])
train_loader = DataLoader(dataset , batch_size= 32 , shuffle=True , collate_fn=custom_collate)

#%% 超参数

encoder_size , decoder_size = tokenizer.get_vocab_size()
encoder_size += len(en)
decoder_size += len(zh)
print(encoder_size , decoder_size)
embedding_size = 128
d_K = 64
d_V = 64
num_heads = 8
padding_index = 0
epoch_num = 5


#%% 训练

model = Transformer(encoder_table_size = encoder_size , decoder_table_size=decoder_size ,
                     embedding_size=embedding_size , d_K=d_K , d_V=d_V , num_heads=num_heads ,
                    encoder_padding_idx = en['<pad>'] , decoder_padding_idx = zh['<pad>'])

loss_fn = nn.CrossEntropyLoss(ignore_index=padding_index)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_values = []

for epoch in range(epoch_num):
    for i , data in enumerate(train_loader):
        x , y , z = data

        output = model(x , y)
        # (B , T , E) -> (B , E , T)
        # crossEntropy 在dim = 1 处计算softmax
        output = output.transpose(-1, -2)

        loss = loss_fn(output, z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f'epoch: {epoch} , step: {i} , loss: {loss.item()}')
            loss_values.append(loss.item())

torch.save(model.state_dict() , './result/model_test.pth')
utils.draw_loss_curve(loss_values)


