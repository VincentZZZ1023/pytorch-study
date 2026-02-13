import torch
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
x_data=torch.Tensor([[1.0],[2.0],[3.0]])
y_data=torch.Tensor([[0],[0],[1]])

class LogisticRegressionModel(torch.nn.Module):
        def __init__(self):
            super(LogisticRegressionModel,self).__init__()
            self.linear=torch.nn.Linear(1,1)
        def forward(self,x):
            y_pred=F.sigmoid(self.linear(x))
            return y_pred
model=LogisticRegressionModel()
criterion=torch.nn.BCELoss(size_average=False)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

loss_list=[]
epoch_list=range(1000)
for epoch in range(1000):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss.item())
    loss_list.append(loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
plt.figure(figsize=(10,6),dpi=120)
plt.plot(epoch_list,loss_list,"b-o",label="train loss",linewidth=2,markersize=3)
plt.title("epoch-loss figure")
plt.xlabel("epoch",fontsize=12)
plt.ylabel("loss",fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()