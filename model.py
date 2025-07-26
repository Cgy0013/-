import torch
import torch.nn as nn

# 多层感知机模型（MLP）
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        """
        :param input_dim: 输入维度（序列长度）
        :param hidden_dim: 隐藏层大小
        """
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),  # 展平成一维
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出单值预测
        )

    def forward(self, x):
        # 输入 x 形状 (batch, seq_len, input_dim) ，这里input_dim通常是1，展平后变 (batch, seq_len*input_dim)
        return self.model(x)

# LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1):
        """
        :param input_dim: 输入特征维度（一般1）
        :param hidden_dim: 隐藏层大小
        :param num_layers: LSTM层数
        """
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # 输入 x 形状 (batch, seq_len, input_dim)
        out, _ = self.lstm(x)  # 输出 (batch, seq_len, hidden_dim)
        out = out[:, -1, :]    # 取序列最后时间步输出
        out = self.fc(out)     # 线性层映射到输出1维
        return out

# Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, model_dim=64, seq_len=20, num_heads=4, num_layers=2, dropout=0.1):
        """
        :param input_dim: 输入特征维度（一般为1）
        :param model_dim: Transformer隐藏层维度
        :param seq_len: 序列长度，用于位置编码
        :param num_heads: 多头注意力头数
        :param num_layers: Transformer编码器层数
        :param dropout: dropout比例
        """
        super(TransformerModel, self).__init__()

        self.input_proj = nn.Linear(input_dim, model_dim)  # 输入线性映射
        # 位置编码参数：形状(1, seq_len, model_dim)，训练参数
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, model_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(model_dim, 1)

    def forward(self, x):
        """
        :param x: 输入张量，形状 (batch, seq_len, input_dim)
        :return: 输出张量，形状 (batch, 1)
        """
        x = self.input_proj(x) + self.pos_embedding[:, :x.size(1), :]  # 添加位置编码
        x = self.transformer(x)  # Transformer编码器输出
        x = x.mean(dim=1)        # 对时间维度求平均 (也可以用最后一步输出)
        out = self.output_layer(x)
        return out
