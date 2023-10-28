import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import psutil

from model.HPM import HPM


def convert_bytes_to_readable(size):
    """将字节大小转换为易读的格式"""
    power = 1024
    n = 0
    power_labels = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size > power:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}B"


# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 使用glob遍历所有CSV文件
csv_files = glob.glob('./dataset/csv/*.csv')

# 读取所有CSV文件并合并为一个大的数据帧
data_frames = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    data_frames.append(df)
data = pd.concat(data_frames)

# 将数据分为训练集和测试集
X = data[['x', 'h', 'dx', 'dh']].values
y = data['time'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据移动到GPU上
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)


model = HPM().to(device)

# 实例化损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型并显示进度条和GPU内存使用情况
num_epochs = 50
batch_size = 4096

for epoch in range(num_epochs):
    train_loss = 0.0
    with tqdm(total=len(X_train_tensor), desc=f"Epoch {epoch+1}") as pbar:
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]

            # 前向传播和计算损失
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)

            # 反向传播和更新权重
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录训练损失、更新进度条和显示GPU内存使用情况
            train_loss += loss.item() * batch_X.size(0)
            pbar.set_postfix(train_loss=train_loss/(i+batch_X.size(0)), gpu_memory= convert_bytes_to_readable(torch.cuda.max_memory_allocated()))
            pbar.update(batch_X.size(0))

    # 在测试集上评估模型并显示进度条和GPU内存使用情况
    test_loss = 0.0
    with torch.no_grad(), tqdm(total=len(X_test_tensor), desc="Test") as pbar:
        for i in range(0, len(X_test_tensor), batch_size):
            batch_X = X_test_tensor[i:i+batch_size]
            batch_y = y_test_tensor[i:i+batch_size]

            # 前向传播和计算损失
            y_pred = model(batch_X)
            loss = criterion(y_pred, batch_y)

            # 记录测试损失、更新进度条和显示GPU内存使用情况
            test_loss += loss.item() * batch_X.size(0)
            pbar.set_postfix(test_loss=test_loss/(i+batch_X.size(0)), gpu_memory=convert_bytes_to_readable(torch.cuda.max_memory_allocated()))
            pbar.update(batch_X.size(0))
            
# 保存模型
torch.save(model.state_dict(), 'my_model.pth')