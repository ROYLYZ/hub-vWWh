"""
数据形状: X: torch.Size([1000, 1]), y: torch.Size([1000, 1])
------------------------------
模型结构:
SinFittingNN(
  (network): Sequential(
    (0): Linear(in_features=1, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=32, bias=True)
    (3): ReLU()
    (4): Linear(in_features=32, out_features=16, bias=True)
    (5): ReLU()
    (6): Linear(in_features=16, out_features=1, bias=True)
  )
)
------------------------------
开始训练...
------------------------------
Epoch [500/3000], Loss: 0.011719
Epoch [1000/3000], Loss: 0.010784
Epoch [1500/3000], Loss: 0.010034
Epoch [2000/3000], Loss: 0.010199
Epoch [2500/3000], Loss: 0.010056
Epoch [3000/3000], Loss: 0.009918

训练完成！
------------------------------
最终损失值: 0.009981
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 生成sin函数数据
X_numpy = np.linspace(-3*np.pi, 3*np.pi, 1000).reshape(-1, 1)
y_numpy = np.sin(X_numpy) + np.random.normal(0, 0.1, X_numpy.shape)  # 添加噪声

# 转换为PyTorch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print(f"数据形状: X: {X.shape}, y: {y.shape}")
print("---" * 10)

# 定义多层神经网络模型
class SinFittingNN(nn.Module):
    def __init__(self):
        super(SinFittingNN, self).__init__()
        # 构建神经网络：输入层 -> 隐藏层1 -> 隐藏层2 -> 输出层
        self.network = nn.Sequential(
            nn.Linear(1, 64),   # 输入层到第一个隐藏层
            nn.ReLU(),          # 激活函数
            nn.Linear(64, 32),  # 隐藏层1到隐藏层2
            nn.ReLU(),
            nn.Linear(32, 16),  # 隐藏层2到隐藏层3
            nn.ReLU(),
            nn.Linear(16, 1)    # 隐藏层3到输出层
        )
        
    def forward(self, x):
        return self.network(x)

# 初始化模型、损失函数和优化器
model = SinFittingNN()
print("模型结构:")
print(model)
print("---" * 10)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 3000
print("开始训练...")
print("---" * 10)

for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)
    
    # 计算损失
    loss = loss_fn(y_pred, y)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 每500个epoch打印一次
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
print("---" * 10)

# 预测结果
with torch.no_grad():
    y_predicted = model(X).numpy()

# 可视化结果
plt.figure(figsize=(12, 6))

# 绘制原始数据和预测结果
plt.scatter(X_numpy, y_numpy, label='Noisy sin(x)', color='blue', alpha=0.3, s=10)
plt.plot(X_numpy, y_predicted, label='Neural Network Prediction', color='red', linewidth=3)
plt.plot(X_numpy, np.sin(X_numpy), label='True sin(x)', color='green', linewidth=2, linestyle='--')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Neural Network Fitting Sin Function')
plt.legend()
plt.grid(True)
plt.show()

# 打印最终损失
with torch.no_grad():
    final_pred = model(X)
    final_loss = loss_fn(final_pred, y)
    print(f"最终损失值: {final_loss.item():.6f}")
