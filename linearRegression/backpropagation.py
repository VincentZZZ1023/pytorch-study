import torch
x_data=[1.0,2.0,3.0]
y_data=[2.0,4.05,5.98]

w=torch.Tensor([1.0])
w.requires_grad=True

def forward(x):
    return x*w
def loss(x,y):
    y_pred=forward(x)
    return (y_pred - y)**2
print ("predict (before training)",4,forward(4).item())
for epoch in range(100):
    for x,y in zip(x_data,y_data):
        loss_pred=loss(x,y)
        loss_pred.backward()
        w.data=w.data-0.01*w.grad.data
        w.grad.data.zero_()
print ("predict (after training)",4,forward(4).item())

print ("验证一下梯度是否累加，看看grad.data.zero_()的作用：）")

grad_list=[]
for x,y in zip(x_data,y_data):
    l=loss(x,y)
    l.backward()
    grad_list.append(w.grad.item())
    w.grad.data.zero_()
w.grad=None
for x,y in zip(x_data,y_data):
    l=loss(x,y)
    l.backward()
print(f"有zero的w的grad:{grad_list}")
print(f"无zero的w的grad:{w.grad.data}")
sum=0
for gradi in grad_list:
    sum=sum+gradi
print(f"有zero的梯度总和:{sum},无zero的grad:{w.grad.data}")