import regex as re
import json
import jieba
# 分词器
class Tokenizer:
    def __init__(self):
        self.merge_items = {}
        self.vocabulary = {i : chr(i) for i in range(128)}
        self.zh_vocab = {}
        self.id_to_char = {}
        self.special_tokens_en = {}
        self.special_tokens_zh = {}
        self.regex = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        self.load_vocabulary()
    def get_vocab_size(self):
        return len(self.vocabulary) , len(self.zh_vocab)

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

    # 添加特殊字符
    def add_special_tokens(self , special_tokens: list):
        start = len(self.vocabulary)
        self.special_tokens_en = {s : i + start for i, s in enumerate(special_tokens)}
        start  = len(self.zh_vocab)
        self.special_tokens_zh = {s : i + start for i, s in enumerate(special_tokens)}
        return self.special_tokens_en , self.special_tokens_zh


    #BPE构建词汇表
    def build_en_vocabulary(self, vocab_size , sentences : list):

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

            print(f"{max_pair} -> {index + i} : {self.vocabulary[index + i]}")
            # 合并
            new_sentences = []
            for code_list in raw_code:
                new_sentences.append(self.__merge(code_list, max_pair, index + i))

            raw_code = new_sentences
            if  i % 1000 == 0:
                self.save_vocabulary()

    # 构建中文词汇表
    def build_zh_vocabulary(self, sentences : list):
        unique_chars = set()

        for strings in sentences:
            tokens = jieba.lcut(strings)
            unique_chars.update(tokens)

        self.zh_vocab = {char: idx for idx, char in enumerate(unique_chars)}
        self.id_to_char = {idx: char for char, idx in self.zh_vocab.items()}

    def build_vocabulary(self, en_sentences : list , vocab_size , zh_sentences : list):
        self.vocabulary = {i: chr(i) for i in range(128)}
        self.merge_items = {}
        self.zh_vocab = {}
        self.id_to_char = {}

        self.build_en_vocabulary(vocab_size , en_sentences)
        self.build_zh_vocabulary(zh_sentences)
        self.save_vocabulary()


    # 保存词汇表
    def save_vocabulary(self, path = 'data.json'):
        dict = {"vocabulary": self.vocabulary,
                "chi_vocab": self.zh_vocab,
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
        self.zh_vocab = dict['chi_vocab']
        self.id_to_char = {v: k for k, v in dict['chi_vocab'].items()}


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

    # 无特殊字符的编码
    def encode_ordinary(self, sentence: str) -> list:
        phrase_list = re.findall(self.regex, sentence)

        code_list = []
        for phrase in phrase_list:
            code_list.extend(self.__encode_phrase(phrase))
        return code_list

    # 允许句子中存在特殊字符
    def encode(self , sentence: str) -> list:
        code_list = []

        # 无特殊字符表
        if not self.special_tokens_en:
            return self.encode_ordinary(sentence)


        # 包含特殊字符表
        # 拆分出句子中的特殊字符
        special_pattern = f"({'|'.join(re.escape(s) for s in self.special_tokens_en.keys() )})"

        sen_chunk = re.split(special_pattern , sentence)

        for chunk in sen_chunk:
            if chunk in self.special_tokens_en:
                code_list.append(self.special_tokens_en[chunk])
            else:
                code_list.extend(self.encode_ordinary(chunk))


        return code_list

    # 中文编码
    def encode_chinese(self, sentence: str) -> list:

        # 无特殊字符表
        if not self.special_tokens_zh:
            return [ord(c) for c in sentence]

        code_list = []
        # 包含特殊字符表
        # 拆分出句子中的特殊字符
        special_pattern = f"({'|'.join(re.escape(s) for s in self.special_tokens_zh.keys())})"

        sen_chunk = re.split(special_pattern, sentence)

        for chunk in sen_chunk:
            if not bool(chunk):
                continue
            if chunk in self.special_tokens_zh:
                code_list.append(self.special_tokens_zh[chunk])
            else:
                code_list.extend([self.zh_vocab[c] for c in jieba.lcut(chunk)])

        return code_list

    # 中文解码
    def decode_chinese(self, tokens: list , flag = True):
        chi = [self.id_to_char[token] for token in tokens if token in self.id_to_char]
        return ''.join(chi) if flag else chi

    # 解码
    def decode(self, tokens: list) -> str:
        res = [self.vocabulary[token] for token in tokens if token in self.vocabulary]

        return ''.join(res)
