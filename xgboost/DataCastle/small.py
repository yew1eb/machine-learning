import xgboost as xgb
import pandas as pd
import numpy as np
import time


def split_data():
    path = 'D:/dataset/RP/'
    small_size = 1000
    dtrain = pd.read_csv(path + 'train_x.csv')
    labels = pd.read_csv(path + 'train_y.csv')
    dtest = pd.read_csv(path + 'test_x.csv')
    dtrain[:small_size].to_csv('./small_data/train_x.csv', index=False)
    labels[:small_size].to_csv('./small_data/train_y.csv', index=False)
    dtest[:small_size].to_csv('./small_data/test_x.csv', index=False)
    return dtrain, labels, dtest


def load_data(dummy=False):
    path = './small_data/'
    dtrain = pd.read_csv(path + 'train_x.csv').iloc[:, 1:]
    labels = pd.read_csv(path + 'train_y.csv').iloc[:, 1:]
    dtest = pd.read_csv(path + 'test_x.csv')
    dtest_uid = dtest.uid
    dtest = dtest.iloc[:, 1:]
    features = pd.read_csv(path + 'features_type.csv')

    if dummy:
        dtrain_dummies = pd.DataFrame()
        dtest_dummies = pd.DataFrame()
        feature_numeric = features.feature[features.type == 'numeric']
        dtrain_numerics = dtrain[feature_numeric]
        dtest_numerics  = dtest[feature_numeric]
        features_category = features.feature[features.type == 'category']  # 总共93个
        features_category = np.ravel(features_category)
        for f in features_category:
            dummy_dtrain = pd.get_dummies(dtrain[f])
            dummy_dtrain = dummy_dtrain.rename(columns=lambda x: str(f) + '_' + str(x))
            dummy_dtest = pd.get_dummies(dtest[f])
            dummy_dtest = dummy_dtest.rename(columns=lambda x: str(f) + '_' + str(x))

            for col in dummy_dtest.columns:
                if col not in dummy_dtrain.columns:
                    dummy_dtrain[col] = np.zeros(dummy_dtrain.shape[0])
            for col in dummy_dtrain.columns:
                if col not in dummy_dtest.columns:
                    dummy_dtest[col] = np.zeros([dummy_dtest.shape[0]])

            dtrain_dummies = pd.concat([dtrain_dummies, dummy_dtrain], axis=1)
            dtest_dummies = pd.concat([dtest_dummies, dummy_dtest], axis=1)

        dtrain = pd.concat([dtrain_numerics, dtrain_dummies], axis=1)
        dtest = pd.concat([dtest_numerics, dtest_dummies], axis=1)

    for i in range(1106, dtrain.shape[1]):
        print(dtrain.columns[i],dtest.columns[i])

    print(dtrain.shape)
    print(dtest.shape)
    return dtrain, labels, dtest, dtest_uid

def main():
    # split_data()
    dtrain, labels, dtest, dtest_uid = load_data(dummy=True)
    xgb_dtrain = xgb.DMatrix(dtrain, label=1-labels.y)
    xgb_dtest  = xgb.DMatrix(dtest)

def fff():
    # setup parameters for xgboost
    param = {}
    param['booster'] = 'gbtree'
    param['objective'] = 'binary:logistic'
    param['scale_pos_weight'] = 8.7
    param['gamma']=0
    param['lambda']= 700
    param['subsample']=0.75
    param['colsample_bytree']=0.30
    param['min_child_weight']=5
    param['max_depth']=8
    param['eta']=0.03
    param['metrics']='auc'

    watchlist = [ (xgb_dtrain,'train')]
    num_round = 10
    bst = xgb.train(param, xgb_dtrain, num_round, watchlist)

    scores = bst.predict(xgb_dtest)
    result = pd.DataFrame({"uid":dtest_uid, "score":scores})
    result.to_csv(str(time.time())+'.csv', index=False)

def model():
    dtrain, labels, dtest = get_data()
    test_uid = dtest.uid
    xgb_dtrain = xgb.DMatrix(dtrain, label=1 - labels.y)
    xgb_dtest = xgb.DMatrix(dtest)

    # setup parameters for xgboost
    param = {}
    param['booster'] = 'gbtree'
    param['objective'] = 'binary:logistic'
    param['scale_pos_weight'] = 8.7
    param['gamma'] = 0
    param['lambda'] = 700
    param['subsample'] = 0.75
    param['colsample_bytree'] = 0.30
    param['min_child_weight'] = 5
    param['max_depth'] = 8
    param['eta'] = 0.03
    param['metrics'] = 'auc'

    watchlist = [(xgb_dtrain, 'train')]
    num_round = 11
    bst = xgb.train(param, xgb_dtrain, num_round, watchlist)

    pred = bst.predict(xgb_dtest)
    result = pd.DataFrame(test_uid)
    result['score'] = pd.Series(pred)
    result.to_csv(str(time.time()) + '.csv', index=False)


if __name__ == '__main__':
    main()
