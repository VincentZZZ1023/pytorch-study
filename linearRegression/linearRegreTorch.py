import torch
import numpy as np 
import matplotlib.pyplot as plt


x_data=torch.tensor([[1.0],[2.0],[3.0]])
y_data=torch.tensor([[3.12],[4.99],[7.08]])
# 检查 Tensor 所在的设备

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear=torch.nn.Linear(1,1)
    def forward(self,x):
        y_pred=self.linear(x)
        return y_pred
model =LinearModel()
criterion=torch.nn.MSELoss(size_average=False)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

print(x_data.device) 

# 检查模型参数所在的设备
print(next(model.parameters()).device)

w_list=[]
loss_list=[]
for epoch in range(1000):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    loss_list.append(loss.item())
    print(epoch,loss)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f"w={model.linear.weight.item()}")
print(f"b={model.linear.bias.item()}")
epoch_list=[]
for i in range(1000):
    epoch_list.append(i+1)
    
plt.figure(figsize=(6,10),dpi=120)
plt.plot(epoch_list,loss_list,"b-o",label='train loss',linewidth=2,markersize=3)
plt.title("epoch loss figure")
plt.xlabel('epoch',fontsize=12)
plt.ylabel('loss',fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
