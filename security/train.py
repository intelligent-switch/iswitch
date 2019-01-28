
import pandas as pd
import numpy
import time
import torch.utils.data as Data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from SecModel import SecModel
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scalar import scaler
import centerloss 
import torch.optim.lr_scheduler as lr_scheduler

torch.manual_seed(1234)

BATCH_SIZE = 300

cuda = torch.cuda.is_available()
print("CUDA: {}".format(cuda))


datatrain = pd.read_csv('train1.csv')
datatrain = numpy.array(datatrain)

xtrain = datatrain[:,:80]
ytrain = datatrain[:,80]
xtrain = preprocessing.StandardScaler().fit_transform(xtrain)
xtrain=torch.from_numpy(xtrain)
ytrain = torch.from_numpy(ytrain)

start = time.clock()
train_dataset = Data.TensorDataset(xtrain, ytrain)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=2)






lr1 = 0.001
lr2=0.5
num_epoch = 150



net = SecModel()
if cuda:
    net = net.cuda()
loss_weight=2
device = torch.device("cuda" if cuda else "cpu")
celoss =  nn.CrossEntropyLoss().cuda()
optimizer4ce = torch.optim.SGD(net.parameters(),lr=lr1,momentum=0.9, weight_decay=0.0005)
sheduler = lr_scheduler.StepLR(optimizer4ce,20,gamma=0.8)

net.train()
for epoch in range(num_epoch):
    total=0
    correct=0
    sheduler.step()
    for step, (xtrain, ytrain) in enumerate(train_loader):
        xtrain = Variable(xtrain.float())
        ytrain = Variable(ytrain.long())
        if cuda: 
            xtrain = xtrain.cuda()
            ytrain = ytrain.cuda()
        feature1,feature2 = net(xtrain)
        _, predicted = torch.max(feature2.data, 1)
        total += xtrain.size(0)
        correct += (predicted.cpu() == ytrain.cpu()).sum()
        loss = celoss(feature2, ytrain)
        optimizer4ce.zero_grad()
        loss.backward()
        optimizer4ce.step()

    print('Epoch [%d/%d] Loss: %.4f'
         % (epoch + 1, num_epoch, loss.item()))

torch.save(net, 'net.pkl')  