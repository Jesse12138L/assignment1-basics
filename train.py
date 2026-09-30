import pickle
import torch
import numpy as np
import torch.nn as nn
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import run_transformer_lm, run_get_batch, run_cross_entropy,run_gradient_clipping, run_get_lr_cosine_schedule,run_save_checkpoint



class TransformerLMWrapper(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.weights = nn.ParameterDict()
        self._key_map = {}   # 下划线key -> 点号key,只建一次

        def add_param(underscore_key, dot_key, tensor):
            self.weights[underscore_key] = nn.Parameter(tensor)
            self._key_map[underscore_key] = dot_key

        add_param("token_embeddings_weight", "token_embeddings.weight", torch.randn(vocab_size, d_model) * 0.02)
        add_param("ln_final_weight", "ln_final.weight", torch.ones(d_model))
        add_param("lm_head_weight", "lm_head.weight", torch.randn(vocab_size, d_model) * 0.02)

        for i in range(num_layers):
            add_param(f"layers_{i}_attn_q_proj_weight", f"layers.{i}.attn.q_proj.weight", torch.randn(d_model, d_model) * 0.02)
            add_param(f"layers_{i}_attn_k_proj_weight", f"layers.{i}.attn.k_proj.weight", torch.randn(d_model, d_model) * 0.02)
            add_param(f"layers_{i}_attn_v_proj_weight", f"layers.{i}.attn.v_proj.weight", torch.randn(d_model, d_model) * 0.02)
            add_param(f"layers_{i}_attn_output_proj_weight", f"layers.{i}.attn.output_proj.weight", torch.randn(d_model, d_model) * 0.02)
            add_param(f"layers_{i}_ln1_weight", f"layers.{i}.ln1.weight", torch.ones(d_model) )
            add_param(f"layers_{i}_ffn_w1_weight", f"layers.{i}.ffn.w1.weight", torch.randn(d_ff, d_model) * 0.02)
            add_param(f"layers_{i}_ffn_w2_weight", f"layers.{i}.ffn.w2.weight", torch.randn(d_model, d_ff) * 0.02)
            add_param(f"layers_{i}_ffn_w3_weight", f"layers.{i}.ffn.w3.weight", torch.randn(d_ff, d_model) * 0.02)
            add_param(f"layers_{i}_ln2_weight", f"layers.{i}.ln2.weight", torch.ones(d_model))
            

    def forward(self, in_indices):
        weights_dict = {self._key_map[k]: v for k, v in self.weights.items()}
        return run_transformer_lm(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            rope_theta=self.rope_theta,
            weights=weights_dict,
            in_indices=in_indices,
        )


# (d_model=256, num_heads=8, num_layers=4, d_ff=1024, context_length=256)。

def train():
    train_data = np.load('data/train_ids.npy', mmap_mode='r')
    val_data = np.load('data/val_ids.npy', mmap_mode='r')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerLMWrapper(10000,256,256,4,8,1024,rope_theta=10000.0).to(device)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9,0.999), eps=1e-8)

    lr=3e-4
    context_length = 256
    max_iters = 5000
    batch_size = 32 
    max_lr_norm = 1.0
    warmup_iters = 500            # 前 500 步线性升温
    min_lr = 3e-5                 # 学习率最低值(通常是 max_lr 的 0.1 倍)

    # 打印/保存
    eval_interval = 500          # 每多少步评估一次验证集 loss
    checkpoint_path = 'data/checkpoint.pt'

    for it in range(max_iters):
        # 1. 取一批数据
        x, y = run_get_batch(train_data, batch_size, context_length, device)

        # 2. 前向 + 计算损失
        logits = model(x)                           # (batch, ctx, vocab)
        loss = run_cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # 3. 反向传播
        optimizer.zero_grad()
        loss.backward()

        lr_now = run_get_lr_cosine_schedule(it, lr, min_lr, warmup_iters, max_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr_now

        run_gradient_clipping(model.parameters(), max_lr_norm)
        optimizer.step()

        # 6. 定期打印和保存
        if it % eval_interval == 0:
            print(f"iter {it}, loss {loss.item():.4f}")
            run_save_checkpoint(model, optimizer,it, checkpoint_path)
