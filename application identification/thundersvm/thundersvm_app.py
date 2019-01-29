from svm import *
import time

y,x = svm_read_problem('svm_app_train.txt')
svm_train(y,x,'svm_app.model','-s 0 -t 2 -g 0.25 -c 10')
y,x=svm_read_problem('svm_app_test.txt')
start = time.clock()

svm_predict(y,x,'svm_app.model','svm_app.predict')


print('testing time: %s Second' % (end - start))
