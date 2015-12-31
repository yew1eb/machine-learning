#/usr/bin/python3


import pandas as pd
import xgboost as xgb
import time
from sklearn.cross_validation import train_test_split

# set data path
train_x_csv = './rp/train_x.csv'
train_y_csv = './rp/train_y.csv'
test_x_csv  = './rp/test_x.csv'
features_type_csv = './rp/features_type.csv'

# load data
train_x = pd.read_csv(train_x_csv)
train_y = pd.read_csv(train_y_csv)
train_xy = pd.merge(train_x, train_y, on='uid')
test = pd.read_csv(test_x_csv)
test_uid = test.uid
test_x = test.drop(['uid'], axis=1)

# split train set,generate train,val,test set
train_xy = train_xy.drop(['uid'], axis=1)
train, val = train_test_split(train_xy, test_size=0.3)
y = train.y
X = train.drop(['y'], axis=1)
val_y = val.y
val_X = val.drop(['y'], axis=1)

# xgboost start here
dtest = xgb.DMatrix(test_x)
dval = xgb.DMatrix(val_X, label=val_y)
dtrain = xgb.DMatrix(X, label=y)

params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'early_stopping_rounds': 100,
    'scale_pos_weight': 0.77,
    'eval_metric': 'auc',
    'gamma': 0.1,
    'min_child_weight': 5,
    'lambda': 700,
    'subsample': 0.7,
    'colsample_bytree': 0.3,
    'max_depth': 8,
    'eta': 0.03,
}

watchlist = [(dval, 'val'), (dtrain, 'train')]
model = xgb.train(params, dtrain, num_boost_round=5000, evals=watchlist)
model.save_model('./rp/xgb.model')

# predict test set (from the best iteration)
test_y = model.predict(dtest, ntree_limit=model.best_ntree_limit)
test_result = pd.DataFrame(columns=['uid', 'score'])
test_result.uid = test_uid
test_result.score = test_y 
test_result.to_csv('xgb_'+str(time.time())+'.csv', index=None, encoding='utf-8')  # remember to edit xgb.csv , add

