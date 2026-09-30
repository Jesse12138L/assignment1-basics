import json
import regex


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):

        self.vocab = vocab
        self.merges = merges
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if special_tokens is None:
            self.special_tokens = []
        else:
            self.special_tokens = special_tokens
        # 建立反向查询字典（根据token查id）
        self.reverse_vocab = {v: k for k, v in vocab.items()}

        self.merge_rank = {pair: rank for rank, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(merges_filepath, encoding="utf-8") as f:
            lines = f.readlines()
        merges = []
        # if作为过滤条件是放在推导式的最右边,跟 for 同级.[处理line的表达式 for line in lines if line.split()]
        merges = [tuple(x.encode("utf-8") for x in line.split()) for line in lines if line.split() ]

        with open(vocab_filepath, encoding="utf-8") as f:
            vocab_raw = json.load(f)
        vocab = {v: k.encode("utf-8") for k, v in vocab_raw.items()}

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if  self.special_tokens: #防special_tokens为空列表
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = '|'.join(f"({regex.escape(t)})" for t in sorted_tokens)
            parts = regex.split(special_pattern, text)
        else:
            parts = [text]

        sen = []
        for p in parts:
            if not p:continue # 防空串
            if p in self.special_tokens:
                sen.append(self.reverse_vocab[p.encode('utf-8')])
            else:
                pre_tokens = regex.findall(self.pattern, p) 
                for tok in pre_tokens:                         # 逐个处理
                    tok = tuple(bytes([b]) for b in tok.encode("utf-8"))
                    while True:
                        best_rank = float('inf')# 无穷大
                        best_pos = -1
                        for i in range(len(tok)-1):
                            pair = (tok[i], tok[i+1])
                            # 字典的 get 方法,查不到就返回"无穷大"(代表这个 pair 不可合并)
                            rank = self.merge_rank.get(pair, float('inf')) 
                            if rank < best_rank:
                                best_rank = rank
                                best_pos = i                                                  
                        # 如果没找到任何可合并的 pair(best_pos 仍是 -1),结束
                        if best_pos == -1:
                            break
                        # 合并 best_pos 位置的两个 token
                        tok = tok[:best_pos] + (tok[best_pos] + tok[best_pos+1],) + tok[best_pos+2:]
                    sen.extend(self.reverse_vocab[t] for t in tok)    
        return sen


    def encode_iterable(self, iterable):
        for text in iterable:
            yield from self.encode(text) # 等价于 for tok in self.encode(text): yield tok


    def decode(self, ids: list[int]) -> str:

        byte_pieces = [self.vocab[id] for id in ids]
        result = b''.join(byte_pieces)   
        return result.decode("utf-8", errors="replace") 