import torch
import torch.nn as nn
import torch.nn.functional as F

in_dim=80
out_dim=15

h1l = 500  
h2l = 400
h3l = 300
h4l = 200
h5l = 100
h6l = 9

class SecModel(nn.Module):
    def __init__(self):
        super(SecModel,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, h1l),nn.BatchNorm1d(h1l),  nn.PReLU())
        self.layer2 = nn.Sequential(nn.Linear(h1l, h2l),nn.BatchNorm1d(h2l),  nn.PReLU())
        self.layer3 = nn.Sequential(nn.Linear(h2l, h3l),nn.BatchNorm1d(h3l),  nn.PReLU())
        self.layer4 = nn.Sequential(nn.Linear(h3l, h4l),nn.BatchNorm1d(h4l),  nn.PReLU())
        self.layer5 = nn.Sequential(nn.Linear(h4l, h5l),nn.BatchNorm1d(h5l),  nn.PReLU())
        self.layer6 = nn.Sequential(nn.Linear(h5l, h6l),nn.BatchNorm1d(h6l),  nn.PReLU())
        self.layer7 = nn.Sequential(nn.Linear(h6l, out_dim,bias=False))
            
    def forward(self, x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.layer5(x)
        x=self.layer6(x)
        x1=x
        x=self.layer7(x)
        return x1,x