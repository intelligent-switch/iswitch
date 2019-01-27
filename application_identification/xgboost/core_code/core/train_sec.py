#!/usr/bin/python

from __future__ import division

import numpy as np
import xgboost as xgb
import pandas as pd

train = pd.read_csv("sec_trainset.csv")
train = np.array(train)

test = pd.read_csv("sec_testset.csv")
test = np.array(test)

train_X = train[:, :80]
train_Y = train[:, 80]

test_X = test[:, :80]
test_Y = test[:, 80]

xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 12
param['tree_method'] = 'gpu_hist'

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 150
bst = xgb.train(param, xg_train, num_round, watchlist)
# get prediction
pred = bst.predict(xg_test)
error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
print('Test error using softmax = {}'.format(error_rate))
bst.save_model('xgb_sec.model')
'''
# do the same thing again, but output probabilities
param['objective'] = 'multi:softprob'
bst = xgb.train(param, xg_train, num_round, watchlist)
# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
pred_prob = bst.predict(xg_test).reshape(test_Y.shape[0], 20)
pred_label = np.argmax(pred_prob, axis=1)
error_rate = np.sum(pred_label != test_Y) / test_Y.shape[0]
print('Test error using softprob = {}'.format(error_rate))
'''
