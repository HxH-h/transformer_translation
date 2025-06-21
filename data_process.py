from PARAMETER import *
import torch

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
import os

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
    # return x , y , z
    for data in zip(*batch):
        # 返回填充后的 batch 数据
        yield pad_sequence([torch.tensor(i) for i in data], batch_first=True, padding_value=PAD_ID)


# 训练英文分词器
if not os.path.exists(EN_MODEL_PREFIX + '.model'):
    spm.SentencePieceTrainer.Train(
        input=RAW_EN_PATH,
        model_prefix=EN_MODEL_PREFIX,
        vocab_size=EN_VOCAB_SIZE,
        model_type='bpe',
        max_sentence_length=10000,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
        num_threads=8,

        unk_piece=UNK,
        bos_piece=BEGIN,
        eos_piece=END,
        pad_piece=PAD,

        unk_id=UNK_ID,
        bos_id=BEGIN_ID,
        eos_id=END_ID,
        pad_id=PAD_ID,
    )
# 训练中文分词器
if not os.path.exists(ZH_MODEL_PREFIX + '.model'):
    spm.SentencePieceTrainer.Train(
        input=RAW_ZH_PATH,
        model_prefix=ZH_MODEL_PREFIX,
        vocab_size=ZH_VOCAB_SIZE,
        model_type='bpe',
        character_coverage = 1.0,
        max_sentence_length=10000,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
        num_threads=8,

        unk_piece=UNK,
        bos_piece=BEGIN,
        eos_piece=END,
        pad_piece=PAD,

        unk_id=UNK_ID,
        bos_id=BEGIN_ID,
        eos_id=END_ID,
        pad_id=PAD_ID,
    )







