"""
SECTION 1 : Load and setup data for training

"""
import pandas as pd
import numpy
import time
import torch.utils.data as Data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from acg_class import ACG
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scalar import scaler

torch.manual_seed(1234)

BATCH_SIZE = 300

# Load
cuda = torch.cuda.is_available()
print("CUDA: {}".format(cuda))


datatrain = pd.read_csv('app_train.csv')
datatrain = numpy.array(datatrain)

xtrain = datatrain[:,:40]
ytrain = datatrain[:,40]
xtrain = preprocessing.StandardScaler().fit_transform(xtrain)

xtrain=torch.from_numpy(xtrain)
ytrain = torch.from_numpy(ytrain)



#put the data into dataloder
start = time.clock()
train_dataset = Data.TensorDataset(xtrain, ytrain)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=2)


"""
SECTION 2 : Build and Train Model

Multilayer perceptron model, with two hidden layer.
input layer : 40 neuron, represents the feature of flow statics
hidden layer : 200/400/600/800/600/400/200 neuron, activation using ReLU
output layer : 20 neuron, represents the class of flows

optimizer = ADAM with 300 batch-size
loss function = categorical cross entropy
learning rate = 0.001
epoch = 150
"""



# Hyperparameters

lr = 0.001
num_epoch = 150


# Build model

net_acg = ACG()
if cuda:
    net_acg = net_acg.cuda()

# Choose optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net_acg.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
loss_history = []

# Train
net_acg.train()
for epoch in range(num_epoch):
    for step, (xtrain, ytrain) in enumerate(train_loader):
        xtrain = Variable(xtrain.float())
        ytrain = Variable(ytrain.long())
        if cuda:
            xtrain = xtrain.cuda()
            ytrain = ytrain.cuda()
        output = net_acg(xtrain)
        loss = criterion(output, ytrain)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [%d/%d] Loss: %.4f'
          % (epoch + 1, num_epoch, loss.data[0]))
    loss_history.append(loss.data[0])

    if epoch % 50 == 0 and epoch != 0:
        lr = lr * 0.95
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    previous_loss = loss.data[0]

    print('learning rate:',lr)

end = time.clock()
print('traning time: %s Seconds' % (end - start))
torch.save(net_acg, 'net_app.pkl')  # save the model

