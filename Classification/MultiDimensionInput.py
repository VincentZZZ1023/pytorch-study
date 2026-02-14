import torch
import pandas as pd 
import numpy as np 
from pathlib import Path
import matplotlib.pyplot as plt
csv_path=Path(__file__).resolve().parent /"diabetes.csv"
df=pd.read_csv(csv_path,header=None)

x_data=torch.tensor(df.iloc[:,:-1].values,dtype=torch.float32)
y_data=torch.tensor(df.iloc[:,-1].values,dtype=torch.float32).view(-1,1)
print(x_data.shape,y_data.shape)

class myModel(torch.nn.Module):
    def __init__(self):
        super(myModel,self).__init__()
        self.linear1=torch.nn.Linear(8,6)
        self.linear2=torch.nn.Linear(6,4)
        self.linear3=torch.nn.Linear(4,1)
        self.relu = torch.nn.ReLU()
        self.sigmoid=torch.nn.Sigmoid()
    def forward(self,x):
        x=self.relu(self.linear1(x))
        x=self.relu(self.linear2(x))
        x=self.sigmoid(self.linear3(x))
        return x
model=myModel()
criterion=torch.nn.BCELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
epochs_list=range(1000)
loss_list=[]

for epoch in range(1000):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss.item())
    loss_list.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.figure(figsize=(10,6),dpi=120)
plt.plot(epochs_list,loss_list,"b-o",label='train loss',linewidth=2,markersize=3)
plt.title("epoch loss figure")
plt.xlabel('epoch',fontsize=12)
plt.ylabel('loss',fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

with torch.no_grad():
    prob = model(x_data)
    pred = (prob >= 0.5).float()
    acc = (pred == y_data).float().mean().item()
print(f"该预测的准确率为: {acc}")