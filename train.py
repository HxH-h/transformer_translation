import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import PARAMETER
from Transformer.Transformer import Transformer
import utils
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu , SmoothingFunction
from PARAMETER import *

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 设置随机数种子
utils.set_seed(PARAMETER.SEED)

#%% 数据处理
# 分词器
en_t , zh_t = utils.load_tokenizer(PARAMETER.EN_MODEL_PREFIX + '.model',
                                   PARAMETER.ZH_MODEL_PREFIX + '.model')

print("分词器加载完成")

#%% 处理数据
data = utils.get_data(PARAMETER.TRAIN_PATH)

tokens = pd.DataFrame()
tokens['en'] = data['en'].apply(lambda x: en_t.Encode(x))
tokens['zh'] = data['zh'].apply(lambda x: zh_t.Encode(x , add_bos=True , add_eos=True))

print("完成分词")

# 创建 dataset 和 dataloader

class Trans_dataset(Dataset):
    def __init__(self , data):
        self.data = data
    def __getitem__(self , index):
        return self.data['en'][index] , self.data['zh'][index][:-1] , self.data['zh'][index][1:]
    def __len__(self):
        return len(self.data)

def custom_collate(batch):
    # batch 中的每个元素是一个样本
    # 提取输入数据并进行填充

    x , y , z = zip(*batch)

    # x = pad_sequence([torch.tensor(i) for i in x], batch_first=True , padding_value=pad_id)
    # y = pad_sequence([torch.tensor(i) for i in y], batch_first=True , padding_value=pad_id)
    # z = pad_sequence([torch.tensor(i) for i in z], batch_first=True , padding_value=pad_id)
    for data in zip(*batch):
        # 返回填充后的 batch 数据
        yield pad_sequence([torch.tensor(i) for i in data], batch_first=True, padding_value=PAD_ID)
    return x , y , z

dataset  = Trans_dataset(tokens)
train_loader = DataLoader(dataset, batch_size= BATCH_SIZE, shuffle=False, collate_fn=custom_collate)



#%% 训练

model = Transformer(encoder_table_size = EN_VOCAB_SIZE, decoder_table_size=ZH_VOCAB_SIZE,
                    embedding_size=EMBEDDING_SIZE, d_K=D_K, d_V=D_V, num_heads=NUM_HEADS,
                    padding_idx = PAD_ID, device=device)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

loss_values = []
bleu_values = []
for epoch in range(EPOCH_NUM):
    for i , data in enumerate(train_loader):
        x , y , z = data

        dev_x = x.to(device)
        dev_y = y.to(device)
        dev_z = z.to(device)

        output = model(dev_x , dev_y)
        # (B , T , E) -> (B , E , T)
        # crossEntropy 在dim = 1 处计算softmax
        output = output.transpose(-1, -2)

        loss = loss_fn(output, dev_z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            # 统计训练损失
            print(f'epoch: {epoch} , step: {i} , loss: {loss.item()}')
            loss_values.append(loss.item())



torch.save(model.state_dict() , PARAMETER.MODEL_PATH)
utils.draw_curve(loss_values , path=PARAMETER.LOSS_PATH)


