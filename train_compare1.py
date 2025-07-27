import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文黑体
plt.rcParams['axes.unicode_minus'] = False    # 负号正常显示
import matplotlib
matplotlib.use('TkAgg')
from data import generate_data
from moirai_moe import MoiraiMoE

# TransformerModel直接用类名
from model import TransformerModel


def plot_mse_bar(mse_transformer, mse_moirai):
    models = ['Transformer', 'Moirai-MoE']
    mses = [mse_transformer, mse_moirai]
    colors = ['skyblue', 'salmon']

    plt.figure(figsize=(6, 5))
    bars = plt.bar(models, mses, color=colors)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.4f}",
                 ha='center', va='bottom', fontsize=12)

    plt.ylabel("均方误差 (MSE)")
    plt.title("Transformer 与 Moirai-MoE 模型预测误差对比")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max(mses) * 1.2)
    plt.show()

def train_model(model, x_train, y_train, x_val, y_val, epochs=30, lr=1e-3):
    criterion = nn.MSELoss() # 使用均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=lr) # Adam优化器学习权重
    model.train()

    for epoch in range(epochs):  # 训练多个 epoch
        optimizer.zero_grad()  # 梯度清零
        outputs = model(x_train)  # 前向传播，得到预测值
        loss = criterion(outputs, y_train)  # 计算训练损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新参数

        # 每 10 个 epoch 或第 1 个 epoch 打印一次验证集性能
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()  # 切换到评估模式（关闭 Dropout 等）
            with torch.no_grad():  # 不计算梯度，加快推理速度、节省内存
                val_pred = model(x_val)  # 验证集上预测
                val_loss = criterion(val_pred, y_val)  # 验证集损失
            print(f"Epoch [{epoch + 1}/{epochs}] Train Loss: {loss.item():.4f} Val Loss: {val_loss.item():.4f}")
            model.train()  # 切换回训练模式
    return model  # 返回训练好的模型


# 测试函数：用于在测试集上评估模型表现
def evaluate_model(model, x_test, y_test):
    model.eval()  # 评估模式
    with torch.no_grad():  # 禁用梯度计算
        preds = model(x_test).squeeze().cpu().numpy()  # 模型预测结果（压缩维度，转为 NumPy）
    targets = y_test.squeeze().cpu().numpy()  # 真实值（压缩维度，转为 NumPy）
    mse = ((preds - targets) ** 2).mean()  # 手动计算均方误差
    return preds, targets, mse  # 返回预测结果、真实值和误差

def plot_results(targets, preds_transformer, preds_moirai):
    plt.figure(figsize=(12,6))
    plt.plot(targets[:100], label='真实值', linewidth=2)
    plt.plot(preds_transformer[:100], label='Transformer预测', linestyle='--')
    plt.plot(preds_moirai[:100], label='Moirai-MoE预测', linestyle='--')
    plt.title("Transformer 与 Moirai-MoE 预测对比（前100个样本）")
    plt.xlabel("样本序号")
    plt.ylabel("数值")
    plt.legend()
    plt.grid(True)
    plt.show()




def main():
    # 1. 生成训练集和测试集数据
    x_train, y_train, x_test, y_test = generate_data(seq_len=20)  # 序列长度为 20
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")  # 打印训练集形状

    # 2. 初始化模型（两个）
    transformer = TransformerModel(input_dim=1, model_dim=64, seq_len=20, num_heads=4, num_layers=2)
    # 创建 Transformer 模型，参数含义如下：
    # input_dim: 每个时间步输入的维度（1维）
    # model_dim: 模型内部表示维度
    # seq_len: 输入序列长度
    # num_heads: 多头注意力的头数
    # num_layers: Transformer 编码器的层数

    moirai = MoiraiMoE(seq_len=20, num_experts=4, hidden_dim=64)
    # 创建 Moirai-MoE 模型，参数含义如下：
    # seq_len: 序列长度
    # num_experts: 专家网络个数
    # hidden_dim: 专家网络内部隐藏层维度

    # 3. 训练模型
    print("训练 Transformer...")
    transformer = train_model(transformer, x_train, y_train, x_test, y_test, epochs=100, lr=1e-3)
    # 训练 Transformer 模型

    print("训练 Moirai-MoE...")
    moirai = train_model(moirai, x_train, y_train, x_test, y_test, epochs=100, lr=1e-3)
    # 训练 Moirai-MoE 模型

    # 4. 测试评估两个模型
    preds_transformer, targets, mse_transformer = evaluate_model(transformer, x_test, y_test)
    # Transformer 的预测结果、真实标签、测试误差

    preds_moirai, _, mse_moirai = evaluate_model(moirai, x_test, y_test)
    # Moirai 的预测结果和误差

    # 打印 MSE 结果
    print(f"Transformer 测试 MSE: {mse_transformer:.6f}")
    print(f"Moirai-MoE 测试 MSE: {mse_moirai:.6f}")

    # 5. 绘制结果对比图
    plot_results(targets, preds_transformer, preds_moirai)
    plot_mse_bar(mse_transformer, mse_moirai)


if __name__ == "__main__":
    main()
