#/usr/bin/python3


import pandas as pd
import xgboost as xgb
import time
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# set data path
path = 'D:/dataset/rp/'
train_x_csv = path+'train_x.csv'
train_y_csv = path+'train_y.csv'
test_x_csv  = path+'test_x.csv'
features_type_csv = path+'features_type.csv'

# load data
train_x = pd.read_csv(train_x_csv)
train_y = pd.read_csv(train_y_csv)
train_xy = pd.merge(train_x, train_y, on='uid')
test = pd.read_csv(test_x_csv)
test_uid = test.uid
test_x = test.drop(['uid'], axis=1)

# split train set,generate train,val,test set
train_xy = train_xy.drop(['uid'], axis=1)
train, val = train_test_split(train_xy, test_size=0.35)
y = train.y
X = train.drop(['y'], axis=1)

def add_data(X,y):
    add_X = pd.read_csv(path+'add_X.csv')
    add_X = add_X.drop(['uid'], axis=1)
    add_y = pd.read_csv(path+'add_y.csv')
    add_y = add_y.drop(['uid'], axis=1)
    add_y = add_y.y
    X = pd.concat([X,add_X], axis=0)
    y = pd.concat([y,add_y], axis=0)
    return X, y

X, y = add_data(X, y)


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
model = xgb.train(params, dtrain, num_boost_round=5, evals=watchlist)
model.save_model('./xgb.model')

# predict test set (from the best iteration)
test_y = model.predict(dtest, ntree_limit=model.best_ntree_limit)
test_result = pd.DataFrame(columns=['uid', 'score'])
test_result.uid = test_uid
test_result.score = test_y 
test_result.to_csv('xgb_'+str(time.time())+'.csv', index=None, encoding='utf-8')  # remember to edit xgb.csv , add

features = model.get_fscore()
features = sorted(features.items(), key=lambda d:d[1])
f_df = pd.DataFrame(features, columns=['feature','fscore'])
f_df.to_csv('./feature_score.csv',index=False)



'''
plt.figure()
import_f = f_df[:10]
import_f.plot(kind='barh', x='feature', y='fscore', legend=False)
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.show()
'''