from tokenizer import Tokenizer
import utils
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
#%% 数据处理
# 获取中英对照数据
data = utils.get_data('./translation2019zh/t.json')

# 构建词汇表
tokenizer = Tokenizer()

en , zh = tokenizer.add_special_tokens(['<begin>','<end>','<pad>'])

# 转token
tokens = pd.DataFrame()
tokens['en'] = data['en'].apply(lambda x: tokenizer.encode(x))
tokens['zh'] = data['zh'].apply(lambda x: tokenizer.encode_chinese('<begin>' + x))
print(len(data))
#%%
# english = tokenizer.decode(tokens.en[0])
# res = tokenizer.decode_chinese(tokens.zh[0])
# print(res)
#%% 自定义 Dataset 和 DataLoader
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

    # x , y , z = zip(*batch)
    #
    # x = pad_sequence([torch.tensor(i) for i in x], batch_first=True)
    # y = pad_sequence([torch.tensor(i) for i in y], batch_first=True)
    # z = pad_sequence([torch.tensor(i) for i in z], batch_first=True)
    for data in zip(*batch):
        # 返回填充后的 batch 数据
        yield pad_sequence([torch.tensor(i) for i in data], batch_first=True , padding_value=zh['<pad>'])


dataset  = Trans_dataset(tokens , zh['<end>'])
train_loader = DataLoader(dataset , batch_size=5 , shuffle=False , collate_fn=custom_collate)

for i , data in enumerate(train_loader):
    x , y , z = data
    print(data)
    print(x)
    print(y)
    print(z)



