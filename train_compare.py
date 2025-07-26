# train_compare.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from model import MLP, LSTM, TransformerModel
from data import generate_data

# -------------------- 超参数 --------------------
SEQ_LEN = 20
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001

# -------------------- 数据准备 --------------------
x_train, y_train, x_test, y_test = generate_data(seq_len=SEQ_LEN)
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=BATCH_SIZE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- 模型初始化 --------------------
models = {
    'MLP': MLP(input_dim=SEQ_LEN),
    'LSTM': LSTM(input_dim=1),
    'Transformer': TransformerModel(input_dim=1, seq_len=SEQ_LEN)
}

# -------------------- 记录数据结构初始化 --------------------
train_losses_all = {}
test_preds_all = {}
test_actuals_all = None
test_mse_all = {}
criterion = nn.MSELoss()

# -------------------- 模型训练与测试 --------------------
for name, model in models.items():
    print(f"\nTraining {name} model...")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_losses = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
    train_losses_all[name] = train_losses

    # 测试阶段
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb)
            preds.extend(pred.cpu().numpy())
            actuals.extend(yb.numpy())
    test_preds_all[name] = preds
    if test_actuals_all is None:
        test_actuals_all = actuals

    # 计算 MSE
    preds_tensor = torch.tensor(np.array(preds))
    actuals_tensor = torch.tensor(np.array(actuals))
    mse = ((preds_tensor - actuals_tensor) ** 2).mean().item()
    test_mse_all[name] = mse
    print(f"{name} Test MSE: {mse:.4f}")

# -------------------- 图像可视化部分 --------------------

# 1. 模型预测 vs 真实值曲线图
plt.figure(figsize=(12, 6))
for name, preds in test_preds_all.items():
    plt.plot(preds, label=f"{name} Prediction")
plt.plot(test_actuals_all, label="Ground Truth", linestyle='dashed', color='black')
plt.legend()
plt.title("Model Predictions vs Ground Truth")
plt.tight_layout()
plt.show()

# 2. 各模型训练损失趋势图
plt.figure(figsize=(8, 5))
for name, losses in train_losses_all.items():
    plt.plot(losses, label=f"{name} Train Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss per Epoch")
plt.legend()
plt.tight_layout()
plt.show()

# 3. 模型测试 MSE 对比柱状图（单独展示）
plt.figure(figsize=(6, 4))
names = list(test_mse_all.keys())
mses = [test_mse_all[n] for n in names]

# 绘制条形图
bars = plt.bar(names, mses, color=['skyblue', 'salmon', 'lightgreen'])

# 在每个柱子顶部显示数值
for bar, mse in zip(bars, mses):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f'{mse:.4f}', ha='center', va='bottom')

plt.ylabel("Test MSE")
plt.title("Test MSE Comparison")
plt.tight_layout()
plt.show()
