import json
import os
import pandas as pd
from opencc import OpenCC
from itertools import chain
from tokenizer import Tokenizer
import matplotlib.pyplot as plt

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)
# 整合数据
def get_data(*path):
    data = []
    for p in path:
        data.append(read_json(p))
    return pd.concat(data, ignore_index=True)
# 合并字符串
def merge_string(data):
    return ''.join(chain.from_iterable(data))
# 拆分字符串
def split_string(text , min_len):
    result = []
    start = 0
    n = len(text)
    punctuations = set(".。")

    while start < n:
        # 当前剩余文本长度小于等于 min_len，则直接加入结果并结束
        if n - start < min_len:
            result.append(text[start:])
            break

        # 查找 [start + min_len, n) 范围内第一个标点符号
        split_pos = -1
        for i in range(start + min_len - 1, n):
            if text[i] in punctuations:
                split_pos = i + 1  # 包括标点本身
                break

        if split_pos != -1:
            result.append(text[start:split_pos])
            start = split_pos
        else:
            # 如果找不到标点，就将剩下的全部加入，并移动到末尾
            result.append(text[start:])
            start = n

    return result



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


def draw_loss_curve(loss_values: list , path = './result/loss.png'):

    # 创建迭代次数的列表，与损失值一一对应
    iterations = list(range(1, len(loss_values) + 1))

    # 绘制折线图
    plt.plot(iterations, loss_values, marker='o')

    # 添加标题和轴标签
    plt.title('Loss Value Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss Value')

    # 显示网格
    plt.grid(True)

    # 展示图表
    plt.savefig(path)



