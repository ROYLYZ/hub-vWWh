import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40


class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


# 定义不同复杂度的模型
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SimpleClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 根据hidden_dims列表构建网络层
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # 添加dropout防止过拟合
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


# 创建数据集
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
output_dim = len(label_to_index)

# 定义要测试的不同模型配置
model_configs = [
    {"name": "单层-小网络", "hidden_dims": [32]},
    {"name": "单层-中网络", "hidden_dims": [128]},
    {"name": "单层-大网络", "hidden_dims": [512]},
    {"name": "两层-小网络", "hidden_dims": [64, 32]},
    {"name": "两层-中网络", "hidden_dims": [256, 128]},
    {"name": "三层网络", "hidden_dims": [256, 128, 64]},
    {"name": "四层网络", "hidden_dims": [512, 256, 128, 64]},
    {"name": "深层窄网络", "hidden_dims": [128, 128, 128, 128, 128]},
    {"name": "宽浅网络", "hidden_dims": [1024]},
]

# 训练不同模型并记录结果
results = []

for config in model_configs:
    print(f"\n{'='*60}")
    print(f"训练模型: {config['name']}")
    print(f"网络结构: {config['hidden_dims']}")
    print(f"总层数: {len(config['hidden_dims']) + 1}")
    print(f"总参数估算: {vocab_size * config['hidden_dims'][0] + sum(config['hidden_dims'][i] * config['hidden_dims'][i+1] for i in range(len(config['hidden_dims'])-1)) + config['hidden_dims'][-1] * output_dim + sum(config['hidden_dims'])}")
    
    # 创建模型
    model = SimpleClassifier(vocab_size, config['hidden_dims'], output_dim)
    
    # 创建数据加载器
    train_size = int(0.8 * len(char_dataset))
    test_size = len(char_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(char_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器
    
    # 训练参数
    num_epochs = 20
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_test_loss = test_loss / len(test_loader)
        accuracy = 100 * correct / total
        
        test_losses.append(avg_test_loss)
        test_accuracies.append(accuracy)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Test Loss: {avg_test_loss:.4f}, "
                  f"Accuracy: {accuracy:.2f}%")
    
    # 保存结果
    results.append({
        "name": config["name"],
        "structure": config["hidden_dims"],
        "train_losses": train_losses,
        "test_losses": test_losses,
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1],
        "final_accuracy": test_accuracies[-1],
        "total_params": sum(p.numel() for p in model.parameters())
    })

# 可视化结果
plt.figure(figsize=(16, 10))

# 1. 训练损失对比
plt.subplot(2, 2, 1)
for i, result in enumerate(results):
    plt.plot(result["train_losses"], label=f"{result['name']}", linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Training Loss Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# 2. 测试损失对比
plt.subplot(2, 2, 2)
for i, result in enumerate(results):
    plt.plot(result["test_losses"], label=f"{result['name']}", linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.title('Test Loss Comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# 3. 最终性能对比（柱状图）
plt.subplot(2, 2, 3)
names = [r["name"] for r in results]
final_test_losses = [r["final_test_loss"] for r in results]
bars = plt.bar(range(len(results)), final_test_losses)
plt.xlabel('Model')
plt.ylabel('Final Test Loss')
plt.title('Final Test Loss Comparison')
plt.xticks(range(len(results)), names, rotation=45, ha='right')
# 为每个柱子添加数值
for bar, loss in zip(bars, final_test_losses):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             f'{loss:.3f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3)

# 4. 准确率对比
plt.subplot(2, 2, 4)
final_accuracies = [r["final_accuracy"] for r in results]
bars = plt.bar(range(len(results)), final_accuracies, color='green')
plt.xlabel('Model')
plt.ylabel('Final Accuracy (%)')
plt.title('Final Accuracy Comparison')
plt.xticks(range(len(results)), names, rotation=45, ha='right')
# 为每个柱子添加数值
for bar, acc in zip(bars, final_accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             f'{acc:.1f}%', ha='center', va='bottom')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 打印详细结果表格
print("\n" + "="*80)
print("模型性能总结:")
print("="*80)
print(f"{'模型名称':<20} {'网络结构':<25} {'参数量':<12} {'最终训练损失':<15} {'最终测试损失':<15} {'最终准确率':<12}")
print("-"*100)

for result in results:
    print(f"{result['name']:<20} {str(result['structure']):<25} "
          f"{result['total_params']:<12} {result['final_train_loss']:<15.4f} "
          f"{result['final_test_loss']:<15.4f} {result['final_accuracy']:<11.2f}%")

# 查找最佳模型
best_model_idx = np.argmin([r["final_test_loss"] for r in results])
best_model = results[best_model_idx]
print("\n" + "="*80)
print(f"最佳模型: {best_model['name']}")
print(f"网络结构: {best_model['structure']}")
print(f"最终测试损失: {best_model['final_test_loss']:.4f}")
print(f"最终准确率: {best_model['final_accuracy']:.2f}%")
print("="*80)

# 保存最佳模型用于后续推理
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

# 重新训练最佳模型用于测试
print("\n使用最佳配置训练最终模型...")
best_config = model_configs[best_model_idx]
final_model = SimpleClassifier(vocab_size, best_config["hidden_dims"], output_dim)
final_criterion = nn.CrossEntropyLoss()
final_optimizer = optim.Adam(final_model.parameters(), lr=0.001)

# 在整个数据集上训练
full_loader = DataLoader(char_dataset, batch_size=32, shuffle=True)
for epoch in range(20):
    final_model.train()
    train_loss = 0.0
    for inputs, labels in full_loader:
        final_optimizer.zero_grad()
        outputs = final_model(inputs)
        loss = final_criterion(outputs, labels)
        loss.backward()
        final_optimizer.step()
        train_loss += loss.item()
    
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch [{epoch+1}/20], Loss: {train_loss/len(full_loader):.4f}")

# 测试最终模型
index_to_label = {i: label for label, i in label_to_index.items()}

test_texts = [
    "帮我导航到北京",
    "查询明天北京的天气",
    "打开音乐播放器",
    "设置闹钟明天七点",
    "打电话给妈妈",
    "查看日历安排",
    "连接wifi网络",
    "搜索附近餐厅"
]

print("\n最终模型测试结果:")
print("-"*50)
for text in test_texts:
    predicted_class = classify_text(text, final_model, char_to_index, vocab_size, max_len, index_to_label)
    print(f"输入: '{text}'")
    print(f"预测: '{predicted_class}'")
    print("-"*30)
