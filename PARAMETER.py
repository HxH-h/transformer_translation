# 路径
RAW_JSON_PATH = './corpus/origin/news.json'
RAW_EN_PATH = './corpus/origin/news_en.txt'
RAW_ZH_PATH = './corpus/origin/news_zh.txt'
EN_MODEL_PREFIX = './result/vocab/en_tokenizor'
ZH_MODEL_PREFIX = './result/vocab/zh_tokenizor'

TRAIN_PATH = './corpus/news_train.json'
TEST_PATH = './corpus/news_test.json'
VAL_PATH = './corpus/news_val.json'

RESULT_PATH = './result/'
LOSS_PATH = RESULT_PATH + 'loss.png'
MODEL_PATH  = RESULT_PATH + 'model.pth'

# 特殊token
BEGIN = '<begin>'
END = '<end>'
PAD = '<pad>'
UNK = '<unk>'
BEGIN_ID = 0
END_ID = 1
PAD_ID = 2
UNK_ID = 3



# 超参数
SEED = 3407
MAX_LEN = 200
EMBEDDING_SIZE = 256
EN_VOCAB_SIZE = 25000
ZH_VOCAB_SIZE = 25000
D_K = 32
D_V = 32
NUM_HEADS = 8
LR = 0.001
BATCH_SIZE = 128
EPOCH_NUM = 20