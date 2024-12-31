import torch
import torch.nn as nn
import pandas as pd

#用GPU训练如果有GPU的话
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# 概述：MINIST数据由5万个图片组成，每个图片是一个784个像素点的灰度图
# 构建两个隐藏层的全连接神经网络，然后将数据2:8开，分为训练数据和测试数据，用以评估模型的精度。
# 最后输出10个概率，分别对应0，1，2，3，4，5，6，7，8，9的概率
#定义网路结构
#[1,784]*[784,512]*[512,444]*[...]*[512,10]
#`输入层` in_channel  *784*, out_channel 随便，譬如256，512，444，我们选444
#`隐藏层1`in_channel  444， out_chanel 512
#`隐藏层2`in_channel  512， out_chanel 512
#`输出层` in_channel  512， out_chanel *10*

# 用Pandas读取MNIST数据
raw_data = pd.read_csv("mnist_train.csv")
raw_data_features = raw_data.iloc[:,1:].values
raw_data_labels = raw_data.iloc[:,0].values

#划分训练和测试数据集
train_f =  raw_data_features[:int(len(raw_data_features)*0.8)]
train_l   =  raw_data_labels[:int(len(raw_data_features)*0.8)]
test_f =  raw_data_features[int(len(raw_data_features)*0.8):]
test_l   =  raw_data_labels[int(len(raw_data_features)*0.8):]

train_features = torch.tensor(train_f).to(torch.float).to(device)
train_labels = torch.tensor(train_l).to(torch.long).to(device)
test_features = torch.tensor(test_f).to(torch.float).to(device)
test_labels = torch.tensor(test_l).to(torch.long).to(device)

#搭建上面模型的全连接网络
model = nn.Sequential(
    nn.Linear(784,444), #计算W，B
    nn.ReLU(), #激活层
    nn.Linear(444,512),
    nn.ReLU(),
    nn.Linear(512,512),
    nn.ReLU(),
    nn.Linear(512,10),
).to(device)

# #定义损失函数 -用交叉商函数
lossfunction = nn.CrossEntropyLoss()
#定义优化器，用以梯度下降计算
optimizer = torch.optim.Adam(params = model.parameters(),lr=0.0001)
#训练100次
for i in range(100):
    optimizer.zero_grad()  #清空优化器梯度（偏导)
    predict = model(train_features)
    loss = lossfunction(predict,train_labels)
    result = torch.argmax(predict,dim=1)
    train_accuracy = torch.mean((result == train_labels).to(torch.float))
    loss.backward() #反向传播
    optimizer.step() #做梯度下降
    print(f"Train {i+1}, Loss: {loss.item()}, accuracy: {train_accuracy.item()}") #损失值

#用测试数据集进行测试评估模型准确性
optimizer.zero_grad()
predict = model(test_features)
result = torch.argmax(predict,dim=1)
test_accuracy = torch.mean((result == test_labels).to(torch.float))
loss = lossfunction(predict,test_labels)
print(f"Test Loss: {loss.item()}, accuracy: {test_accuracy.item()}") #损失值

#保存训练好的模型完整模型，包括了模型结构和W和b权重参数
torch.save(model,"model.pth")
print("训练完成")
