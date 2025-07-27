import torch
import matplotlib.pyplot as plt
from data import generate_data
from model import TransformerModel
from moirai_moe import MoiraiMoE

def predict(model, x_test):
    model.eval()
    with torch.no_grad():
        preds = model(x_test).squeeze().numpy()
    return preds

if __name__ == "__main__":
    # 1. 生成数据
    x_train, y_train, x_test, y_test = generate_data(seq_len=20)
    y_test_np = y_test.squeeze().numpy()

    # 2. 加载模型（训练后的模型或新实例）
    transformer = TransformerModel(seq_len=20)
    moirai = MoiraiMoE(seq_len=20, num_experts=4, hidden_dim=64)

    # 3. 加载已训练模型参数（如果训练后保存了）
    # transformer.load_state_dict(torch.load("transformer.pt"))
    # moirai.load_state_dict(torch.load("moirai.pt"))

    # 4. 加载训练完的模型或重新训练（建议先在 train.py 训练再来这步）
    transformer_preds = predict(transformer, x_test)
    moirai_preds = predict(moirai, x_test)

    # 5. 可视化对比前 100 个样本
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_np[:100], label="True", linewidth=2)
    plt.plot(transformer_preds[:100], label="Transformer", linestyle='--')
    plt.plot(moirai_preds[:100], label="Moirai-MoE", linestyle='--')
    plt.title("预测值对比（前100个样本）")
    plt.xlabel("样本索引")
    plt.ylabel("目标值")
    plt.legend()
    plt.grid(True)
    plt.show()
