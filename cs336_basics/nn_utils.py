import torch
import math
import os
import numpy as np
from einops import einsum
from einops import rearrange
from jaxtyping import Float,Bool,Int
from torch import Tensor

def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    values, _ = torch.max(in_features, dim=dim, keepdim=True)
    in_features -= values
    e = torch.exp(in_features) 
    out = e / torch.sum(e,dim=dim, keepdim=True)
    return out


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:

    return in_features * torch.sigmoid(in_features)


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:

    out = einsum(in_features, weights, "... d_in, d_out d_in -> ... d_out")

    return out


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids,
) -> Float[Tensor, " ... d_model"]:

    # 根据weight查对应ids的向量，将ids对应的id转换成向量
    return weights[token_ids]


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    # (batch_size, q ,d_model),(batch_size, k ,d_model),(batch_size, v ,d_model)
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) /(d_k ** 0.5)
    if mask is not None:
       scores = scores.masked_fill(~mask, float("-inf"))# mask把scores对应False的位置填上-inf
    probs = run_softmax(scores,dim=-1)
    return probs @ V


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:

    # 一维数组，存了 D/2 个旋转速度。下标 i=0 时值最大（转得快），下标 i=D/2-1 时值最小（转得慢）
    freqs = theta ** (-2*torch.arange(d_k // 2, device=in_query_or_key.device) / d_k)
    # 对于序列里第 pos 个 token，它对应的旋转角度就是 pos * freqs。比如位置 3 的低频维度，角度就是 3 * 0.0001
    angles = token_positions.unsqueeze(-1) * freqs
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    # 分离奇偶
    x_even = in_query_or_key[..., 0::2]
    x_odd = in_query_or_key[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd  = x_even * sin + x_odd * cos
    out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
    return out 


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:

    Q = run_linear(d_model,d_model,q_proj_weight,in_features)#(...,s,d)
    K = run_linear(d_model,d_model,k_proj_weight,in_features)
    V = run_linear(d_model,d_model,v_proj_weight,in_features)

    Q = rearrange(Q, "... seq (heads hidden) -> ... heads seq hidden", heads=num_heads)
    K = rearrange(K, "... seq (heads hidden) -> ... heads seq hidden", heads=num_heads)
    V = rearrange(V, "... seq (heads hidden) -> ... heads seq hidden", heads=num_heads)

    seq_len = in_features.shape[-2]
    mask=torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool,device=in_features.device), diagonal=1)

    atten = run_scaled_dot_product_attention(Q,K,V,~mask)

    atten = rearrange(atten, "... heads seq hidden -> ... seq (heads hidden)", heads=num_heads)
    
    return run_linear(d_model,d_model,o_proj_weight,atten)



def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:

    Q = run_linear(d_model,d_model,q_proj_weight,in_features)#(...,s,d)
    K = run_linear(d_model,d_model,k_proj_weight,in_features)
    V = run_linear(d_model,d_model,v_proj_weight,in_features)

    Q = rearrange(Q, "... seq (heads hidden) -> ... heads seq hidden", heads=num_heads)
    K = rearrange(K, "... seq (heads hidden) -> ... heads seq hidden", heads=num_heads)
    V = rearrange(V, "... seq (heads hidden) -> ... heads seq hidden", heads=num_heads)

    d_head = d_model // num_heads
    Q=run_rope(d_head,theta,max_seq_len,Q,token_positions)
    K=run_rope(d_head,theta,max_seq_len,K,token_positions)

    seq_len = in_features.shape[-2]
    mask=torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool,device=in_features.device), diagonal=1)

    atten = run_scaled_dot_product_attention(Q,K,V,~mask)

    atten = rearrange(atten, "... heads seq hidden -> ... seq (heads hidden)", heads=num_heads)
    
    return run_linear(d_model,d_model,o_proj_weight,atten)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:

    x = torch.mean(in_features**2,dim=-1,keepdim=True) + eps

    return (in_features / torch.sqrt(x)) * weights


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    x1 = run_silu(run_linear(d_model,d_ff,w1_weight,in_features))
    x2 = run_linear(d_model,d_ff,w3_weight,in_features)
    return run_linear(d_ff, d_model, w2_weight, x1 * x2)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:

    seq_len = in_features.shape[-2]
    token_pos = torch.arange(seq_len, device=in_features.device).unsqueeze(0)

    # input(b,s,d)
    x1 = run_rmsnorm(d_model,1e-5,weights['ln1.weight'],in_features)
    atten = run_multihead_self_attention_with_rope(d_model,
                                                    num_heads,
                                                    max_seq_len,
                                                    theta,
                                                    weights['attn.q_proj.weight'],
                                                    weights['attn.k_proj.weight'],
                                                    weights['attn.v_proj.weight'],
                                                    weights['attn.output_proj.weight'],
                                                    x1,
                                                    token_pos)
    in_features=atten+in_features

    x2 = run_rmsnorm(d_model,1e-5,weights['ln2.weight'],in_features)
    x2 = run_swiglu(d_model,d_ff,
                    weights['ffn.w1.weight'],
                    weights['ffn.w2.weight'],
                    weights['ffn.w3.weight'],
                    x2)
    in_features=x2+in_features

    return in_features


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:

    # (b,s)->(b,s,d)
    x = run_embedding(vocab_size,d_model,weights['token_embeddings.weight'],in_indices)

    for i in range(num_layers):
        # 去掉key前面的 layers.{i}.才是block对应的key
        prefix = f"layers.{i}."
        block_weights = {}
        for key ,value in weights.items():
            if key.startswith(prefix):
                block_weights[key[len(prefix):]] = value

        x = run_transformer_block(d_model,
                                    num_heads,
                                    d_ff,
                                    context_length,
                                    rope_theta,
                                    block_weights,
                                    x)

    x = run_rmsnorm(d_model,1e-5,weights['ln_final.weight'],x)

    logits = run_linear(d_model,vocab_size,weights['lm_head.weight'], x)

    return logits #（b,s,v）



def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:

    val,_ = torch.max(inputs,dim=-1,keepdim=True)
    inputs = inputs-val

    log_probs = inputs - torch.logsumexp(inputs, dim=-1, keepdim=True)  # "log(sum(exp(x)))"
    loss = -log_probs[range(inputs.shape[0]), targets]# 取出对应target（标签位置）的值
    return loss.mean()

# 梯度裁减，利用l2范式比较（平方和开根）
def run_gradient_clipping(parameters, max_l2_norm: float) -> None:
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            total_norm += param.grad.pow(2).sum()
    total_norm = total_norm ** 0.5

    if total_norm > max_l2_norm:
        for param in parameters:
            if param.grad is not None:
                param.grad *= max_l2_norm / total_norm

# 余弦退火lr调度器
def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        lr = (it / warmup_iters) * max_learning_rate
    else:
        progress = min((it - warmup_iters) / (cosine_cycle_iters - warmup_iters), 1.0)
        lr = min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * progress))
    return lr   


def run_get_batch(
    dataset, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:

    # 输入片段:dataset[start : start+context_length]
    # 标签片段:dataset[start+1 : start+context_length+1]
    # 即最大到 len(dataset) - context_length - 1
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)

    input, label = [], []
    for start in starts:
        input.append(dataset[start:start+context_length])
        label.append(dataset[start+1:start+context_length+1])

    input = np.stack(input)
    label = np.stack(label)

    out1 = torch.from_numpy(input).long().to(device)
    out2 = torch.from_numpy(label).long().to(device)

    return (out1,out2)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out,
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)



def run_load_checkpoint(
    src,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
    