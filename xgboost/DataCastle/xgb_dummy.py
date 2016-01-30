#!/usr/bin/env python
# -*- coding:utf-8 -*-

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import time


def split_data():
    path = 'D:/dataset/rp/'
    small_size = 1000
    dtrain = pd.read_csv(path + 'train_x.csv')
    labels = pd.read_csv(path + 'train_y.csv')
    dtest = pd.read_csv(path + 'test_x.csv')

    dtrain[:small_size].to_csv(path+'small_data/train_x.csv', index=False)
    labels[:small_size].to_csv(path+'small_data/train_y.csv', index=False)
    dtest[:small_size].to_csv(path+'small_data/test_x.csv', index=False)
    return dtrain, labels, dtest

def load_data(dummy=False):
    path = 'D:/dataset/rp/small_data/'
    #path = 'D:/dataset/rp/'
    train_x = pd.read_csv(path + 'train_x.csv')
    train_x = train_x.drop(['uid'], axis=1)

    train_y = pd.read_csv(path + 'train_y.csv')
    train_y = train_y.drop(['uid'], axis=1)

    test_x = pd.read_csv(path + 'test_x.csv')
    test_uid = test_x.uid
    test_x = test_x.drop(['uid'], axis=1)

    if dummy: # 将分类类型的变量转为哑变量
        features = pd.read_csv(path + 'features_type.csv')
        features_category = features.feature[features.type == 'category']
        encoded = pd.get_dummies(pd.concat([train_x, test_x], axis=0), columns=features_category)
        train_rows = train_x.shape[0]
        train_x = encoded.iloc[:train_rows, :]
        test_x  = encoded.iloc[train_rows:, :]

    return train_x, train_y, test_x, test_uid

def main():
    train_x, train_y, test_x, test_uid = load_data(dummy=True)

    # 交叉验证，分割训练数据集
    random_seed = 10
    X_train, X_val, y_train, y_val= train_test_split(train_x, train_y, test_size=0.33, random_state=2016)
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_val   = xgb.DMatrix(X_val, label=y_val)
    xgb_test  = xgb.DMatrix(test_x)

    # 设置xgboost分类器参数
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'early_stopping_rounds': 100,
        'scale_pos_weight': 0.77,
        'gamma': 0.1,
        'min_child_weight': 5,
        'lambda': 700,
        'subsample': 0.7,
        'colsample_bytree': 0.3,
        'max_depth': 8,
        'eta': 0.03,
        'nthread': 4
    }
    watchlist = [(xgb_val, 'test'), (xgb_train, 'train')]
    num_round = 10
    bst = xgb.train(params, xgb_train, num_boost_round=num_round, evals=watchlist)
    bst.save_model('./xgb.model')

    scores = bst.predict(xgb_test, ntree_limit=bst.best_ntree_limit)
    result = pd.DataFrame({"uid":test_uid, "score":scores}, columns=['uid','score'])
    result.to_csv('dummy_'+str(time.time())+'.csv', index=False)


    features = bst.get_fscore()
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


if __name__ == '__main__':
    main()
