import numpy as np
import scipy.sparse
import time
import xgboost as xgb
import pandas as pd
from sklearn.metrics import classification_report
from sklearn import metrics

bst = xgb.Booster(model_file='xgb_app.model')

test = pd.read_csv("acg_test.csv")
test = np.array(test)


test_X = test[:, :40]
test_Y = test[:, 40]

xg_test = xgb.DMatrix(test_X, label=test_Y)

start = time.clock()

for i in range(38):

    pred = bst.predict(xg_test)
    

#end = time.clock()
#print('testing time: %s Seconds' % (end - start))



    error_rate = np.sum(pred != test_Y) / test_Y.shape[0]

print('Test error using softmax = {}'.format(error_rate))
end = time.clock()
print('testing time: %s Seconds' % (end - start))
fpr, tpr, thresholds = metrics.roc_curve(test_Y,pred, pos_label=2)
auc = metrics.auc(fpr, tpr)
print(classification_report(test_Y,pred),fpr,tpr,auc)

