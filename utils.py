import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from collections import OrderedDict
import numpy as np
import random
import sentencepiece as spm
def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)
def save_json(df , path):
    data = df.to_dict(orient='records')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
# 整合数据
def get_data(*path):
    data = []
    for p in path:
        data.append(read_json(p))
    return pd.concat(data, ignore_index=True)

# 划分数据集
def split_data(df):
    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(
        df['en'], df['zh'], test_size=0.2, random_state=42)

    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)

    # 合并成 DataFrame
    train_df = pd.DataFrame({'en': X_train, 'zh': y_train})
    test_df = pd.DataFrame({'en': X_test, 'zh': y_test})
    val_df = pd.DataFrame({'en': X_val, 'zh': y_val})

    save_json(train_df, './corpus/news_train.json')
    save_json(test_df, './corpus/news_test.json')
    save_json(val_df, './corpus/news_val.json')


# tsv 转 json
def tsv_to_json(path , output_path):
    df = pd.read_csv(path, sep='\t' , header= None , names=['en' , 'zh'])

    # 将 DataFrame 转换为 JSON 格式
    data = df.to_dict(orient='records')

    # 保存为 JSON 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# json 转 txt
def json_to_txt(df , key , path):
    # 将 'text' 列转为 NumPy 数组并保存
    np.savetxt(path, df[key].values, fmt='%s', encoding='utf-8')


# 繁体转简体
def convert_to_simplified(p , key):
    cc = OpenCC('t2s')

    with open(p, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for i in range(len(data)):
        para = []
        for str in data[i][key]:
            para.append(cc.convert(str))
        data[i][key] = para


    with open(p, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4 , ensure_ascii=False)


def draw_curve(values: list , path = './result/loss.png'):

    # 创建迭代次数的列表，与损失值一一对应
    iterations = list(range(1, len(values) + 1))

    plt.figure(figsize = (20, 10))

    # 绘制折线图
    plt.plot(iterations, values, marker='o')

    # 添加标题和轴标签
    plt.title('Loss Value Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss Value')

    # 显示网格
    plt.grid(True)

    # 展示图表
    plt.savefig(path)
    plt.close()

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 加载分词器
def load_tokenizer(en_path , zh_path):
    en_tokenizer = spm.SentencePieceProcessor()
    en_tokenizer.Load(en_path)

    zh_tokenizer = spm.SentencePieceProcessor()
    zh_tokenizer.Load(zh_path)
    return en_tokenizer , zh_tokenizer


# 多GPU训练，转换为单GPU/CPU
def load_model_safely(model, path, map_location='cpu'):

    state_dict = torch.load(path, map_location=map_location)

    # 检查是否为 DataParallel 保存的模型（带 "module." 前缀）
    if any(k.startswith("module.") for k in state_dict.keys()):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v
        state_dict = new_state_dict

    # 加载 state_dict
    model.load_state_dict(state_dict)
    return model


