import json
import pandas as pd
import regex as re

# 读取json文件
path = './translation2019zh/translation_valid.json'
def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

data = read_json(path)

# 创建分词器
class Tokenizer:
    def __init__(self):
        self.merge_items = {}
        self.vocabulary = {i : chr(i) for i in range(128)}
        self.regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # 获取相邻字符对 和 出现次数
    def __get_pairs(self, code_list: list , pairs: dict):
        for pair in zip(code_list, code_list[1:]):
            pairs[pair] = pairs.get(pair, 0) + 1


    # 合并字符对
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


    #BPE构建词汇表
    def build_vocabulary(self, vocab_size , sentences : list):

        # 获取句子初始的字符编码列表
        raw_code = [ list(s.encode('utf-8')) for sentence in sentences for s in re.findall(self.regex, sentence)]

        index = len(self.vocabulary)
        merge_num = vocab_size - index

        # BPE
        for i in range(merge_num):
            # 统计相邻字符对 出现的次数
            pairs = {}
            for code_list in raw_code:
                self.__get_pairs(code_list, pairs)

            # 取最大
            max_pair = max(pairs, key=lambda x: pairs[x])
            # 记录合并项 更新词汇表
            self.merge_items[max_pair] = index + i
            self.vocabulary[index + i] = self.vocabulary[max_pair[0]] + self.vocabulary[max_pair[1]]
            # 合并
            new_sentences = []
            for code_list in raw_code:
                new_sentences.append(self.__merge(code_list, max_pair, index + i))

            raw_code = new_sentences



    # 保存词汇表
    def save_vocabulary(self, path = 'data.json'):
        dict = {"vocabulary": self.vocabulary,
                "merge":  {str(k): v for k, v in self.merge_items.items()}}
        # 将字典保存为JSON文件
        with open(path, 'w', encoding='utf-8') as json_file:
            json.dump(dict, json_file, ensure_ascii=False, indent=4)

    # 加载词汇表
    def load_vocabulary(self, path = 'data.json'):
        # 从JSON文件加载字典
        with open(path, 'r', encoding='utf-8') as json_file:
            dict = json.load(json_file)
        # 类型转换
        self.vocabulary = {int(k): v for k, v in dict['vocabulary'].items()}
        self.merge_items = {eval(k) : int(v) for k, v in dict['merge'].items()}


    # 对每个正则的分词进行编码
    def __encode_phrase(self, phrase: str) -> list:
        code_list = list(phrase.encode('utf-8'))

        while len(code_list) > 1:
            pairs = {}
            self.__get_pairs(code_list, pairs)
            # 找对应的索引值最小的 字符对 进行合并
            pair = min(pairs, key=lambda x: self.merge_items.get(x, float('inf')))
            if pair not in self.merge_items:
                break
            code_list = self.__merge(code_list, pair, self.merge_items[pair])

        return code_list

    # 编码
    def encode(self, sentence: str) -> list:
        phrase_list = re.findall(self.regex, sentence)
        code_list = []
        for phrase in phrase_list:
            code_list.extend(self.__encode_phrase(phrase))
        return code_list




    # 解码
    def decode(self, tokens: list) -> str:
        for i in range(len(tokens)):
            tokens[i] = self.vocabulary[tokens[i]]
        return ''.join(tokens)




tokenizer = Tokenizer()

tokenizer.build_vocabulary(150, data['english'].head(10).to_list())
code_list = tokenizer.encode('he is at 123 or ;,.')
print(len('he is at 123 or ;,.'))
print(len(code_list))
print(tokenizer.decode(code_list))
tokenizer.save_vocabulary()
tokenizer.load_vocabulary()


