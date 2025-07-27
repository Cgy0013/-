import numpy as np
import torch

def generate_data(seq_len=20, total_samples=1000):
    """
    生成带噪声的正弦波时间序列数据
    :param seq_len: 输入序列长度
    :param total_samples: 样本数量
    :return: x_train, y_train, x_test, y_test 四个Tensor，均已转float32并带batch维度
    """
    #np.linspace(0, 100, total_samples + seq_len)：
    # 在 0 到 100 之间等间距生成 total_samples + seq_len 个点，作为横坐标（时间轴）。
    x = np.linspace(0, 100, total_samples + seq_len)
    #正态分布噪声（均值为0，标准差为0 .1），模拟现实中不稳定因素
    y = np.sin(x) + np.random.normal(scale=0.1, size=len(x))
    #X：存储所有输入序列（长度为 seq_len）。
    #Y：存储所有对应的目标值（即序列之后的那个点）
    X, Y = [], []
    for i in range(total_samples):
        # 对于每一个样本位置 i，提取一个长度为 seq_len 的序列作为输入
        X.append(y[i:i + seq_len])
        # 该序列的下一个时间点的值作为预测目标（单值回归）
        Y.append(y[i + seq_len])

    # 将列表转换为 NumPy 数组
    X = np.array(X)  # shape: (samples, seq_len)
    Y = np.array(Y)  # shape: (samples,)
    # 加一个特征维度，变成 (samples, seq_len, 1)，适用于 RNN/LSTM/Transformer 输入格式
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # shape: (samples, seq_len, 1)
    Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)  # shape: (samples, 1)
    # 按 80% / 20% 划分训练集和测试集
    train_size = int(0.8 * total_samples)
    # 切分出训练集和测试集
    x_train, y_train = X[:train_size], Y[:train_size]  # 训练集
    x_test, y_test = X[train_size:], Y[train_size:]  # 测试集

    return x_train, y_train, x_test, y_test
if __name__ == "__main__":
    # 简单验证数据形状是否正确，并可视化前几个序列
    import matplotlib.pyplot as plt
    x_train, y_train, x_test, y_test = generate_data()
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    # 可视化前10个训练样本及其标签点
    for i in range(3):
        plt.plot(range(20), x_train[i].squeeze(), label=f"Input seq {i}")
        plt.scatter(20, y_train[i].item(), c='r', label=f"Target {i}")
    plt.title("示例时间序列输入 + 预测目标")
    plt.legend()
    plt.show()