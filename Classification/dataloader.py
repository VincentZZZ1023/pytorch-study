import torch
import pandas as pd 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path

csv_path=Path(__file__).resolve().parent /'diabetes.csv'

class DiabetesDataset(Dataset):
    def __init__(self,filepath):
        df=pd.read_csv(filepath,header= None)
        self.len=df.shape[0]
        self.x_data=torch.tensor(df.iloc[:,:-1].values,dtype=torch.float32)
        self.y_data=torch.tensor(df.iloc[:,-1].values,dtype=torch.float32).view(-1,1)
        
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.len
    




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
def main():
    dataset=DiabetesDataset(csv_path)
    train_loader=DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)
    model=myModel()
    criterion=torch.nn.BCELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
    epochs_list=range(1000)
    loss_list=[]


    for epoch in range(100):
        for i,data in enumerate(train_loader,0):
            inputs,labels=data
            
            y_pred=model(inputs)
            loss=criterion(y_pred,labels)
            # print(epoch,i,loss.item())
            if i%10==0:
                print(epoch,i,loss.item())
                loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
if __name__ == "__main__":
    main()