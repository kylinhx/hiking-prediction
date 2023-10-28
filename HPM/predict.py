import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from model.HPM import HPM

# 读取CSV文件
data = pd.read_csv('./dataset/csv/2022-02-12向阳口珍珠湖避静寺环穿.csv')
X_new = data[['x', 'h', 'dx', 'dh']].values
y = data['time'].values

# 实例化模型并加载保存的参数
model = HPM()
model.load_state_dict(torch.load('my_model.pth'))

# 对新数据进行预测
with torch.no_grad():
    X_new_tensor = torch.tensor(X_new, dtype=torch.float32)
    y_new_pred = model(X_new_tensor).numpy()

plt.plot(y, label='True')
plt.plot(y_new_pred, label='Predicted')
plt.legend()
plt.show()
# 输出预测值
print(y_new_pred)