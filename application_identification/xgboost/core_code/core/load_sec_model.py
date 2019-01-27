import numpy as np
import scipy.sparse
import time
import xgboost as xgb
import pandas as pd
from sklearn.metrics import classification_report

bst = xgb.Booster(model_file='xgb_sec.model')

test = pd.read_csv("sec_testset.csv")
test = np.array(test)


test_X = test[:, :80]
test_Y = test[:, 80]

xg_test = xgb.DMatrix(test_X, label=test_Y)
print(xg_test)
start = time.clock()

for i in range(1):

    pred = bst.predict(xg_test)
    

end = time.clock()
print('testing time: %s Seconds' % (end - start))


for i in range(2):

    pred = bst.predict(xg_test)

end = time.clock()
print('testing time: %s Seconds' % (end - start))


for i in range(4):

    pred = bst.predict(xg_test)

end = time.clock()
print('testing time: %s Seconds' % (end - start))


for i in range(6):

    pred = bst.predict(xg_test)

end = time.clock()
print('testing time: %s Seconds' % (end - start))


for i in range(8):

    pred = bst.predict(xg_test)

end = time.clock()
print('testing time: %s Seconds' % (end - start))

for i in range(10):

    pred = bst.predict(xg_test)

end = time.clock()
print('testing time: %s Seconds' % (end - start))

print(classification_report(test_Y,pred))
error_rate = np.sum(pred != test_Y) / test_Y.shape[0]

print('Test error using softmax = {}'.format(error_rate))
