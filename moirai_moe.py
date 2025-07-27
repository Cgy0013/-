import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """
    单个专家网络：用于学习时间序列的某种子模式
    可根据需要用 LSTM / GRU 替代这里的 MLP
    输入维度，隐藏层维度
    """
    def __init__(self, input_dim, hidden_dim):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),  # 将输入展平为 (batch_size, seq_len)送入全连接层
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出一个预测值
        )

    def forward(self, x):
        return self.net(x)


class GatingNetwork(nn.Module):
    """
    门控网络：根据输入序列自适应分配专家权重
    输出维度为专家数量，并通过 Softmax 转换为概率
    """
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),  # 统一输入形状
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)  # 转成概率
        )

    def forward(self, x):
        return self.net(x)  # 返回每个专家的权重 (batch_size, num_experts)


class MoiraiMoE(nn.Module):
    """
    Moirai-MoE 主模型：融合多个专家的输出，加权组合
    seq_len：输入时间序列的长度（即一个样本有多少个时间点）num_experts：专家数量，默认为 4
    hidden_dim：每个专家内部 MLP 的隐藏层大小
    """
    def __init__(self, seq_len, num_experts=4, hidden_dim=64):
        super(MoiraiMoE, self).__init__()
        self.num_experts = num_experts
        #用 nn.ModuleList 创建一个包含 num_experts 个 MLP 专家 的列表。
        #每个专家结构相同，但参数独立。
        #Expert(seq_len, hidden_dim) 输入是展平的序列，输出是一个单点预测值。
        self.experts = nn.ModuleList([Expert(seq_len, hidden_dim) for _ in range(num_experts)])
        self.gating = GatingNetwork(seq_len, num_experts)

    def forward(self, x):
        # x: (batch_size, seq_len, 1)
        batch_size = x.size(0)

        # 1. 获取门控输出：shape = (batch_size, num_experts)
        gate_weights = self.gating(x)  # softmax 概率分布

        # 2. 获取所有专家的预测结果：每个 expert 输出 shape = (batch_size, 1)
        # 对同一个输入 x，每个专家都单独计算一个输出
        # 用 stack 把这些专家输出拼在一起，变成一个三维张量
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # shape = (batch_size, num_experts, 1)

        # 3. 加权求和专家输出（batch-wise）
        # 将门控权重扩展成 (batch_size, num_experts, 1)，
        # 以便和 expert_outputs 相乘（逐专家加权）
        # 然后 sum(dim=1)，对所有专家求加权和 ⇒ 得到最终预测输出

        gate_weights = gate_weights.unsqueeze(-1)  # (batch_size, num_experts, 1)
        out = torch.sum(gate_weights * expert_outputs, dim=1)  # (batch_size, 1)

        return out
