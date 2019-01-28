import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn import preprocessing



datatrain = pd.read_csv('train1.csv')
datatrain = np.array(datatrain)
xtrain = datatrain[:,:80]
ytrain = datatrain[:,80]
scaler = preprocessing.StandardScaler().fit(xtrain)
