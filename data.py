import numpy as np
import torch

def generate_data(seq_len=20, total_samples=1000):
    """
    生成带噪声的正弦波时间序列数据
    :param seq_len: 输入序列长度
    :param total_samples: 样本数量
    :return: x_train, y_train, x_test, y_test 四个Tensor，均已转float32并带batch维度
    """
    x = np.linspace(0, 100, total_samples + seq_len)
    y = np.sin(x) + np.random.normal(scale=0.1, size=len(x))

    X, Y = [], []
    for i in range(total_samples):
        X.append(y[i:i+seq_len])
        Y.append(y[i+seq_len])

    X = np.array(X)
    Y = np.array(Y)

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (samples, seq_len, 1)
    Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)  # (samples, 1)

    train_size = int(0.8 * total_samples)
    x_train, y_train = X[:train_size], Y[:train_size]
    x_test, y_test = X[train_size:], Y[train_size:]

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