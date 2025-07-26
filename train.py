import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from model import MLP, LSTM, TransformerModel
from data import generate_data
from sklearn.metrics import mean_squared_error

# 超参数配置
SEQ_LEN = 20
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001

# 选择模型（可改为 'MLP' 或 'LSTM'）
MODEL_TYPE = 'MLP'

# 生成数据
x_train, y_train, x_test, y_test = generate_data(seq_len=SEQ_LEN)

# 数据加载器
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=BATCH_SIZE)

# 模型实例化（参数名和 model.py 保持一致）
if MODEL_TYPE == 'MLP':
    model = MLP(input_dim=SEQ_LEN)
elif MODEL_TYPE == 'LSTM':
    model = LSTM(input_dim=1)
elif MODEL_TYPE == 'Transformer':
    model = TransformerModel(input_dim=1, seq_len=SEQ_LEN)
else:
    raise ValueError("Unsupported MODEL_TYPE")

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 训练过程
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
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# 测试过程
model.eval()
preds, actuals = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        pred = model(xb)
        preds.extend(pred.cpu().numpy())
        actuals.extend(yb.numpy())
# 计算并打印测试集均方误差 MSE
mse = mean_squared_error(actuals, preds)
print(f"Test MSE: {mse:.4f}")

# 绘制预测与真实曲线
plt.figure(figsize=(10, 4))
plt.plot(preds, label='Prediction')
plt.plot(actuals, label='Ground Truth')
plt.legend()
plt.title(f"{MODEL_TYPE} Prediction vs Ground Truth")
plt.show()

# 绘制训练损失曲线
plt.figure()
plt.plot(train_losses)
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()
