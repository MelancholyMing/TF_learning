import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import variable

sns.set(style='whitegrid')
plt.rcParams["patch.force_edgecolor"] = True

m = 2  # slope
c = 3  # intercept

x = np.random.rand(256)

noise = np.random.randn(256) / 4

y = x * m + c + noise

df = pd.DataFrame()

df['x'] = x
df['y'] = y
print(df)
sns.lmplot(x='x', y='y', data=df)
plt.show()

x_train = x.reshape(-1, 1).astype('float32')
y_train = y.reshape(-1, 1).astype("float32")
x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train)
# x_train = torch.from_numpy(x_train)
# y_train = torch.from_numpy(y_train)


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


input_dim = x_train.shape[1]
output_dim = y_train.shape[1]
print(input_dim)
model = LinearRegressionModel(input_dim, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epoch_list = []
loss_list = []
for epoch in range(1000):
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    epoch_list.append(epoch)
    loss_list.append(loss.item())
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# print(model.parameters())
[w, b] = model.parameters()


# print(w.data)
# print(b.data)
# print('=========')
# print(model.linear.weight.item())
# print(model.linear.bias.item())

def plot_current_fit(title=""):
    plt.figure(figsize=(12, 4))
    plt.title(title)
    plt.scatter(x, y, s=8)
    w1 = w.data[0][0]
    b1 = b.data[0]
    x1 = np.array([0., 1.])
    y1 = x1 * w1.numpy() + b1.numpy()
    plt.plot(x1, y1, 'r', label="current fit {:.3f},{:.3f}".format(w1, b1))
    plt.xlabel("x(input)")
    plt.ylabel("y(target)")
    plt.legend()
    plt.show()

plot_current_fit("before training")