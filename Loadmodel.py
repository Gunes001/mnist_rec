from pickle import FALSE
import torch
import torch.nn as nn
import pandas as pd


# 检查是否有可用的 MPS 设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 直接加载整个模型，并设置 weights_only=False
model = torch.load("model.pth",weights_only=False).to(device)
model.eval()

# 用Pandas读取MNIST数据abs
raw_data = pd.read_csv("mnist_test.csv")
raw_data_features = raw_data.iloc[:,1:].values
raw_data_labels = raw_data.iloc[:,0].values

test_features = torch.tensor(raw_data_features[200:2000]).to(torch.float).to(device)
test_labels = torch.tensor(raw_data_labels[200:2000]).to(torch.long).to(device)

#预测
with torch.no_grad():
    predict = model(test_features)
    result = torch.argmax(predict,dim=1)

print(test_labels)
print(result)
