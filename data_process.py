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
        self.vocabulary = {i : chr(i) for i in range(128)}

    # 获取相邻字符对 和 出现次数
    def __get_pairs(self, code_list: list) -> dict:
        pairs = {}
        for pair in zip(code_list, code_list[1:]):
            pairs[pair] = pairs.get(pair, 0) + 1

        return pairs
    def __merge(self , code_list , pair , index) -> list:
        code_len = len(code_list)
        if code_len == 1:
            return code_list

        new_code_list = []
        i = 0
        while i < code_len:
            if i < code_len - 1 and code_list[i] == pair[0] and code_list[i + 1] == pair[1]:
                new_code_list.append(index)
                i += 2
            else:
                new_code_list.append(code_list[i])
                i += 1
        return new_code_list



    #TODO BPE构建词汇表
    def build_vocabulary(self, sentences : list):
        # 获取句子初始的字符编码列表
        raw_code = [ord(char) for sentence in sentences for char in sentence]

        index = len(self.vocabulary)
        merge_num = self.vocab_size - index

        # BPE
        for i in range(merge_num):
            # 统计相邻字符对 出现的次数
            pairs = self.__get_pairs(raw_code)
            # 取最大
            max_pair = max(pairs, key=lambda x: pairs[x])
            self.vocabulary[index + i] = self.vocabulary[max_pair[0]] + self.vocabulary[max_pair[1]]
            # 合并
            raw_code = self.__merge(raw_code, max_pair, index + i)




    def encode(self, sentence):
        pass
    def decode(self, tokens):
        pass




tokenizer = Tokenizer(vocab_size=150, sentences=data['english'].head(10).to_list)

tokenizer.build_vocabulary(data['english'].head(2).to_list())


