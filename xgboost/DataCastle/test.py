import xgboost as xgb
import pandas as pd
import numpy as np
from scipy import sparse


data = np.random.rand(5,10) # 5 entities, each contains 10 features
test = np.random.rand(5,10)
label = np.random.randint(2, size=5) # binary target
dtrain = xgb.DMatrix(data, label=label)
dtest = xgb.DMatrix(test)

param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }


# specify validations set to watch performance
watchlist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 2
bst = xgb.train(param, dtrain, num_round, watchlist)

# this is prediction
preds = bst.predict(dtest)
labels = dtest.get_label()
print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))
bst.save_model('0001.model')