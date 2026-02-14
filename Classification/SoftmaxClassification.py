#MNIST用crossEntropy来分类，这里的crossEntropy为:softmax+NLLloss

import torch
from torch.utils.data import dataset
from torch.utils.data import Dataloader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F 
import torch.optim as optim


def main():
    batch_size=64
    transforms=transforms.Compose([transforms.ToTensor(),transforms.Nomalize()])

if __name__=="__main__":
    main()