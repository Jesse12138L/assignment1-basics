
import pickle
import numpy as np
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.bpe import train_bpe


def main():
    with open('data/tinystories_50m.txt', encoding='utf-8') as f:
        text = f.read()
    vocab, merges = train_bpe('data/tinystories_50m.txt',10000,["<|endoftext|>"])
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    sen = tokenizer.encode(text)

    with open('data/tokenizer_vocab_merges.pkl', 'wb') as f:
        pickle.dump({'vocab': vocab, 'merges': merges}, f) #保存bpe，避免每次都要重新训练

    sen_array = np.array(sen, dtype=np.uint16)
    np.save('data/tinystories_50m_ids.npy', sen_array)# 保存切好的sen

    data = np.load('data/tinystories_50m_ids.npy')
    n = len(data)
    split = int(n * 0.95)
    train_data = data[:split]#切训练和验证
    val_data = data[split:]
    np.save('data/train_ids.npy', train_data)
    np.save('data/val_ids.npy', val_data)


if __name__ == "__main__":
    main()