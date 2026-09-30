
import os
import regex
from collections import Counter


def get_pairs(word_tuple):
    '''获取key(tuple)的i和i+1作为一对pair存入列表中返回'''
    # input: ex: tuple:(b'w', b'i', b'd', b'e', b's', b't')
    # output list[(b'w', b'i'),(b'i', b'd'),...]
    l = []
    for i in range(len(word_tuple)-1):
        l.append((word_tuple[i],word_tuple[i+1]))
    return l    

def merge_all_words(words, winner):
    '''合并pair(winner),并记录更改过后的pair及其frequent'''

    new_words = {}
    change = {}
    for (key ,value) in words.items():
        i = 0
        result = []
        while i < len(key):
            if i+1<len(key) and (key[i], key[i+1]) == winner:
                result.append(key[i]+key[i+1])
                i+=2 # 跳过被合并的i+1,从i+2开始
            else:
                result.append(key[i])
                i+=1
        new_tuple = tuple(result)
        if new_tuple != key:
            change[key] = (new_tuple,value) # {old,(new,fre)}
        new_words[new_tuple] = new_words.get(new_tuple, 0) + value
    return new_words,change

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    '''文本 → UTF-8 编码 → 一串字节(每个 0-255) → BPE 在字节上操作。'''
    
    # 开文件获取文本
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    # 将文本（由多个篇章组成）中的special_tokens去掉，形成多个篇章
    parts = [text]
    for s in special_tokens:
        new_parts = []
        for part in parts:
            new_parts.extend(part.split(s))
        parts = new_parts    

    # 正则化，gpt2预分词处理
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens = []
    for par in parts:
        if not par:
            continue
        tokens.extend(regex.findall(pattern, par))# 每个篇章分别按照规则分词并组成tokens

    # 按频率计数对预分词排名
    counts = Counter(tokens)

    #key = ' iron'  字符串,5个字符: 空格, i, r, o, n
    #key.encode("utf-8")  → b' iron' 5个字节: [32, 105, 114, 111, 110]
    # 第2步: for b in ... 遍历每个字节
    # b = 32   空格
    # b = 105  i
    # b = 114  r
    # b = 111  o
    # b = 110  n
    # # 第3步: bytes([b]) 把每个整数包成单字节 bytes
    # bytes([32])   # → b' '
    # bytes([105])  # → b'i'
    # bytes([114])  # → b'r'
    # bytes([111])  # → b'o'
    # bytes([110])  # → b'n'
    # # 第4步: tuple(...) 包成元组
    # (b' ', b'i', b'r', b'o', b'n')
    words = {}
    for key,fre in counts.items():
        # UTF-8 在这里的作用是把文本变成字节序列
        words[tuple(bytes([b]) for b in key.encode("utf-8"))] = fre

    # 初始化词表:0-255 的单字节 token(共256个)
    # 初始词表的 0-255 并不依赖 UTF-8——它就是"所有可能的单字节值"。UTF-8 是编码方式,词表是字节值本身。
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])

    # 将special_tokens接入后续词表
    for index,s in enumerate(special_tokens):
        vocab[index+256] = s.encode("utf-8")


    merges = []
    next_id = 256 + len(special_tokens)
    # 创建counter便于后面字典+=value操作
    Counts = Counter({})
    # 初始第一次先遍历所有words找到所有pair并记录frequent（没有就新增，有就累加），存入counts中
    for (key, value) in words.items():
        for pair in get_pairs(key):
            Counts[pair] += value

    # 循环合并
    while len(vocab) < vocab_size:

        max_val = max(Counts.values()) # 找fre最大的pair 
        #若有相同的最大fre，按 tuple 字典序比较(先比第一个元素,相等再比第二个)
        winner = max([k for k, v in Counts.items() if v == max_val])
        merges.append(winner)
        vocab[next_id] = winner[0]+winner[1]# 将合并的编码记录到词表中
        next_id+=1
        words, changes = merge_all_words(words, winner)# 获取合并后的words和修改记录

        # 根据修改记录，不需要重新遍历words找winner，只需要根据新旧记录找到对应pair的索引进行增量操作
        # 旧tuple:(b'w', b'i', b'd', b'e', b's', b't') —— 合并前的样子
        # 新tuple:(b'w', b'i', b'd', b'e', b'st') —— 合并后的样子
        # 减旧 pair(×3): (w,i)-3、(i,d)-3、(d,e)-3、(e,s)-3、(s,t)-3
        # 加新 pair(×3): (w,i)+3、(i,d)+3、(d,e)+3、(e,st)+3
        # es被自动减去了
        for old,(new_tuple,fre) in changes.items():
            for pair in get_pairs(old):
                Counts[pair]-=fre
            for pair in get_pairs(new_tuple):
                Counts[pair]+=fre

        if len(vocab) % 1000== 0: 
            print(f"{len(vocab)}/{vocab_size}, merges so far: {len(merges)}")

                
    return vocab, merges
    