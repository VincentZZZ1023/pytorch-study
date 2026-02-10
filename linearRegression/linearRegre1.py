import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
x_data=[1.0, 2.0, 3.0]
y_data=[3.0, 5.0, 7.0]
def forward(x,w,b):
    return x*w+b
def loss(x,w,b, y):
    y_hat=forward(x,w,b)
    return (y_hat-y)**2 
w_list=[]
b_list=[]
loss_list=[]
best_index=0
loss_min=1000000000
for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(0.0,4.1,0.1):
        w_list.append(w)
        b_list.append(b)
        l_sum=0
        for x_ori,y_ori in zip(x_data,y_data):
            y_pre=forward(x_ori,w,b)
            loss_pre=loss(x_ori,w,b,y_ori)
            l_sum=l_sum+loss_pre    
        l_sum=l_sum/3
        loss_list.append(l_sum)
        if l_sum<loss_min :
            loss_min=l_sum
            best_index=len(loss_list)-1
print(f"best w b and loss:w:{w_list[best_index]:.6f} ,b:{b_list[best_index]:.6f} ,loss:{loss_list[best_index]:.6f}")
fig=plt.figure()
ax =fig.add_subplot(projection="3d")
ax.plot_trisurf(w_list, b_list, loss_list, cmap='rainbow')
plt.show() 