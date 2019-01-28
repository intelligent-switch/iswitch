import pandas as pd
import torch
import time
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from SecModel import SecModel
from scalar import scaler
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

torch.manual_seed(1234)

cuda = torch.cuda.is_available()
print("CUDA: {}".format(cuda))

BATCH_SIZE = 200

restore_net = SecModel()
restore_net = torch.load('net.pkl')
if cuda:
    restore_net = restore_net.cuda()
restore_net.eval()



datatest = pd.read_csv('test1.csv')



datatest = np.array(datatest)
xtest = datatest[:,:80]

ytest = datatest[:,80]
xtest = scaler.transform(xtest)



xtest = torch.from_numpy(xtest)

ytest=torch.from_numpy(ytest)



test_dataset = Data.TensorDataset(xtest, ytest)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=2)





total=0
correct=0
i=0
for step, (xtest, ytest) in enumerate(test_loader):
    xtest = Variable(xtest.float())
    if cuda:
        xtest = xtest.cuda()
    ytest = ytest.long()
    feature,out = restore_net(xtest)
    _, predicted = torch.max(out.data, 1)
    total += xtest.size(0)
    correct += (predicted.cpu() == ytest).sum()
    if i==0:
            label=np.array(ytest.cpu().data)
            predicts=np.array(predicted.cpu().data)
    else:
            label = np.array(np.concatenate((label,np.array(ytest.cpu().data)),axis=0))
            predicts = np.array(np.concatenate((predicts,np.array(predicted.cpu().data)),axis=0))
    i=i+1

print(precision_score(label,predicts, average=None))
print(recall_score(label,predicts, average=None))