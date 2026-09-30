
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)



    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue   # 没有梯度就跳过(和梯度裁剪里一样)

                grad = param.grad

                # 1. 取或初始化 state
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)

                state["step"] += 1
                m = state["m"]
                v = state["v"]

                # 2. 权重衰减(先对参数本身做,和梯度无关)
                param.data -= lr * weight_decay * param.data

                # 3. 更新一阶矩、二阶矩（原地操作，更省内存）
                # 等价于m = beta1 * m + (1 - beta1) * grad
                # v = beta2 * v + (1 - beta2) * grad ** 2
                # state["m"] = m  
                # state["v"] = v
                m.mul_(beta1).add_(grad, alpha=1 - beta1)          # m = β1*m + (1-β1)*g
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) # v = β2*v + (1-β2)*g²

                # 4. 偏置修正 + 参数更新
                t = state["step"]
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                param.data -= lr * m_hat / (torch.sqrt(v_hat) + eps)

