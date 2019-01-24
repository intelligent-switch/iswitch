import torch
import torch.nn as nn
import torch.nn.functional as F

h1l = 200
h2l = 400
h3l = 600
h4l = 800
h5l = 600
h6l = 400
h7l = 200

class ACG(nn.Module):

    def __init__(self):
        super(ACG,self).__init__()
        self.fc1 = nn.Linear(40, h1l)
        self.fc2= nn.Linear(h1l, h2l)
        self.fc3 = nn.Linear(h2l, h3l)
        self.fc4 = nn.Linear(h3l, h4l)
        self.fc5 = nn.Linear(h4l, h5l)
        self.fc6 = nn.Linear(h5l, h6l)
        self.fc7 = nn.Linear(h6l, h7l)
        self.fc8 = nn.Linear(h7l, 67)
        self.bn1 = nn.BatchNorm1d(200)
        self.bn2 = nn.BatchNorm1d(400)
        self.bn3 = nn.BatchNorm1d(600)
        self.bn4 = nn.BatchNorm1d(600)
        self.bn5 = nn.BatchNorm1d(600)
        self.bn6 = nn.BatchNorm1d(400)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.bn1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.bn2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.bn3(x))
        x = F.relu(self.fc4(x))
        #x = F.relu(self.drop(x))
        #x = F.relu(self.bn4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.bn5(x))
        x = F.relu(self.fc6(x))
        #x = F.relu(self.bn6(x))
        x = F.relu(self.fc7(x))
        #x = F.relu(self.fc8(x))
        x = self.fc8(x)
        return x
