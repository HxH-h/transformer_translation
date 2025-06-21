import torch.nn as nn
import torch
from torch.utils.data import DataLoader


import PARAMETER
from Transformer.Transformer import Transformer
import utils
import pandas as pd

from PARAMETER import *
from data_process import Trans_dataset , custom_collate
import gc

# 检查设备
flag = torch.cuda.is_available()
device = torch.device('cuda' if flag else 'cpu')

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

tokens = tokens[
    (tokens['en'].str.len() <= MAX_LEN) &
    (tokens['zh'].str.len() <= MAX_LEN)
].reset_index(drop=True)
print(len(tokens))
print("完成分词")



dataset  = Trans_dataset(tokens)
train_loader = DataLoader(dataset, batch_size= BATCH_SIZE, shuffle=True, collate_fn=custom_collate)



#%% 训练

model = Transformer(encoder_table_size = EN_VOCAB_SIZE, decoder_table_size=ZH_VOCAB_SIZE,
                    embedding_size=EMBEDDING_SIZE, d_K=D_K, d_V=D_V, num_heads=NUM_HEADS,
                    padding_idx = PAD_ID, device=device)
if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 块 GPU")
    model = nn.DataParallel(model)
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
        torch.cuda.empty_cache()
        if i % 100 == 0:
            # 统计训练损失
            print(f'epoch: {epoch} , step: {i} , loss: {loss.item()}')

            loss_values.append(loss.item())

    torch.cuda.empty_cache()
    gc.collect()



torch.save(model.state_dict() , PARAMETER.MODEL_PATH)
utils.draw_curve(loss_values , path=PARAMETER.LOSS_PATH)



