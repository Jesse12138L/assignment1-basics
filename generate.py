import torch
from cs336_basics.tokenizer import Tokenizer



def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_p=0.95):

    ids = tokenizer.encode(prompt)
    ids = torch.tensor([ids])
    with torch.no_grad():
        for i in range(max_new_tokens):
            logits = model(ids)
            # 只需要预测“下一个词”
            last_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(last_logits, dim=-1)

            # 对所有词按概率从高到低排序，并计算前缀累积和（例如 [0.5, 0.3, 0.2] -> [0.5, 0.8, 1.0]）
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)

            # top-p 截断:累积概率超过 top_p 之后的词,概率设为 0
            cutoff = cumulative >= top_p                # 布尔 mask,超过 top_p 的位置为 True
            cutoff[..., 0] = False # 安全栓。强制把掩码的第一个位置设为 False（保留）（cutoff 的形状是 [1, vocab_size]），即保留第一个维度的0号

            #masked_fill：将 cutoff 为 True 的位置（低概率尾部）直接置为 0.0。
            sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)

            # 重新归一化(砍掉低概率词后,剩下的概率加起来不再是1)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            # 采样:从截断后的分布里随机抽一个下标(位置)
            idx = torch.multinomial(sorted_probs, num_samples=1)   # 形状 (1, 1)

            # 把"位置下标"映射回"真实的 vocab 下标"
            token_id = torch.gather(sorted_indices, -1, idx).item()   # 标量 int
                        # 把新 token 追加到 ids,并做滑动窗口截断
            ids = torch.cat([ids, torch.tensor([[token_id]])], dim=1)
            if ids.shape[1] > model.context_length:
                ids = ids[:, -model.context_length:]

            # 如果生成了 <|endoftext|>,提前停止
            if token_id == tokenizer.encode("<|endoftext|>")[0]:
                break

        return tokenizer.decode(ids[0].tolist())


if __name__ == "__main__":
    import pickle
    from train import TransformerLMWrapper

    # 加载 checkpoint
    ckpt = torch.load('data/checkpoint.pt', map_location='cuda')
    model = TransformerLMWrapper(10000, 256, 256, 4, 8, 1024, rope_theta=10000.0).to('cuda')
    model.load_state_dict(ckpt['model'])

    # 加载 tokenizer
    with open('data/tokenizer_vocab_merges.pkl', 'rb') as f:
        saved = pickle.load(f)
    tokenizer = Tokenizer(saved['vocab'], saved['merges'], special_tokens=['<|endoftext|>'])

    text = generate(model, tokenizer, "Hello, i am Jesse", max_new_tokens=20)
    print(text)