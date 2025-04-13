import json
import pandas as pd


# 读取json文件
path = './translation2019zh/translation_valid.json'
def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

data = read_json(path)


# 创建分词器
class Tokenizer:
    def __init__(self, vocab_size , sentences: list):
        self.vocab_size = vocab_size
        self.vocabulary = {chr(i) : i for i in range(128)}

    # 获取相邻字符对 和 出现次数
    def __get_pairs(self, code_list: list) -> dict:
        pairs = {}
        for pair in zip(code_list, code_list[1:]):
            pairs[pair] = pairs.get(pair, 0) + 1

        return pairs



    #TODO BPE构建词汇表
    def build_vocabulary(self, sentences : list):
        # 获取句子初始的字符编码列表
        raw_code = [ord(char) for sentence in sentences for char in sentence]

        # 统计相邻字符对 出现的次数
        pairs = self.__get_pairs(raw_code)
        # 取最大
        max_pair = max(pairs, key=lambda x : pairs[x])
        print(max_pair , pairs[max_pair])


    def encode(self, sentence):
        return [self.vocabulary[char] for char in sentence]
    def decode(self, tokens):
        pass




tokenizer = Tokenizer(vocab_size=10000, sentences=data['english'].head(10).to_list)

print(tokenizer.build_vocabulary(data['english'].head(2).to_list()))