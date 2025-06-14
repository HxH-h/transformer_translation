import PARAMETER
from tokenizer import Tokenizer
import utils
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
import os

# 训练英文分词器
if not os.path.exists(PARAMETER.EN_MODEL_PREFIX + '.model'):
    spm.SentencePieceTrainer.Train(
        input=PARAMETER.RAW_EN_PATH,
        model_prefix=PARAMETER.EN_MODEL_PREFIX,
        vocab_size=PARAMETER.EN_VOCAB_SIZE,
        model_type='bpe',
        max_sentence_length=10000,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
        num_threads=8,

        unk_piece=PARAMETER.UNK,
        bos_piece=PARAMETER.BEGIN,
        eos_piece=PARAMETER.END,
        pad_piece=PARAMETER.PAD,

        unk_id=PARAMETER.UNK_ID,
        bos_id=PARAMETER.BEGIN_ID,
        eos_id=PARAMETER.END_ID,
        pad_id=PARAMETER.PAD_ID,
    )
# 训练中文分词器
if not os.path.exists(PARAMETER.ZH_MODEL_PREFIX):
    spm.SentencePieceTrainer.Train(
        input=PARAMETER.RAW_ZH_PATH,
        model_prefix=PARAMETER.ZH_MODEL_PREFIX,
        vocab_size=PARAMETER.ZH_VOCAB_SIZE,
        model_type='bpe',
        character_coverage = 1.0,
        max_sentence_length=10000,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
        num_threads=8,

        unk_piece=PARAMETER.UNK,
        bos_piece=PARAMETER.BEGIN,
        eos_piece=PARAMETER.END,
        pad_piece=PARAMETER.PAD,

        unk_id=PARAMETER.UNK_ID,
        bos_id=PARAMETER.BEGIN_ID,
        eos_id=PARAMETER.END_ID,
        pad_id=PARAMETER.PAD_ID,
    )








