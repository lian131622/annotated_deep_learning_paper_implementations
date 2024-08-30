import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable

# 生成数据
t = np.linspace(0, 100, 1000)
sin1 = np.sin(0.2 * t)
sin2 = np.sin(0.5 * t)
combined_signal = sin1 + sin2

# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
combined_signal = scaler.fit_transform(combined_signal.reshape(-1, 1))


# 创建时间序列数据集
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)


time_step = 10
X, Y = create_dataset(combined_signal, time_step)

# 将数据转换为 PyTorch 张量
X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).float()

# reshape 输入为 [samples, time steps, features] 的格式
X = X.reshape(X.shape[0], X.shape[1], 1)


# 构建 LSTM 模型
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions


model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 100
for i in range(epochs):
    for seq, labels in zip(X, Y):
        optimizer.zero_grad()
        y_pred = model(seq.view(1, time_step, -1))
        single_loss = loss_function(y_pred, labels.view(1, -1))
        single_loss.backward()
        optimizer.step()
    if i % 10 == 0:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

# 预测
with torch.no_grad():
    train_predict = model(X).data.numpy()

# 反向缩放数据
train_predict = scaler.inverse_transform(train_predict)
Y = scaler.inverse_transform(Y.view(-1, 1).numpy())

# 可视化结果
plt.plot(t[:-time_step - 1], Y, label='True Signal')
plt.plot(t[:-time_step - 1], train_predict, label='Predicted Signal')
plt.legend()
plt.show()

print('end')
