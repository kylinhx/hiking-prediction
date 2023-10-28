# 北京理工大学 校级大创 徒步时间预测小程序开发项目
### 参与人：韩昫、胡子骄、王子昊、苗姚、李祉颐

## 一、更新日志
### 1、2024年5月2日初始化仓库
```
HPM
|
data
|   |
|   dataloader.py   dataset.py
|
model
|   |
|   HPM.py  models.py
utils
    |
    loss.py plot.py
```
dataloader.py:  读取数据

dataset.py:     简历数据集实例

HPM.py:         HPM模型文件

models.py:      各种网路模块

loss.py:        计算损失函数等

plot:           画图、保存结果

utiles.py       对kml文件进行处理，并将其转换成csv文件格式

### 1、2024年5月5日对代码进行修改和增删

修改utiles.py，对原始的kml文件进行读取以及处理，在指定文件夹下生成对应的csv文件

完成初步的train、predict文件编写

### 2、2024年10月27日增加前端展示，修改代码