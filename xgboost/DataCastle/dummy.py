import xgboost as xgb
import pandas as pd
import numpy as np
import time
from sklearn.cross_validation import train_test_split

def load_data(dummy=False):
    path = './rp/'
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
        features_category = features.feature[features.type == 'category']
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
        dtest = dtest[dtrain.columns]

    return dtrain, labels, dtest, dtest_uid

def main():
    #split_data()
    dtrain, labels, dtest, dtest_uid = load_data(dummy=True)
    xgb_dtrain = xgb.DMatrix(dtrain, label=labels.y)
    xgb_dtest  = xgb.DMatrix(dtest)
    random_seed = 233
    X_train, X_test, y_train, y_test = \
		train_test_split(dtrain, labels, test_size=0.3, random_state=random_seed)
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_test  = xgb.DMatrix(X_test, label=y_test)
    # setup parameters for xgboost
    param = {}
    param['seed'] = random_seed
    param['booster'] = 'gbtree'
    param['objective'] = 'binary:logistic'
    param['scale_pos_weight'] = 2.1
    param['gamma']=0
    param['lambda']= 700
    param['subsample']=0.7
    param['colsample_bytree']=0.30
    param['min_child_weight']=5
    param['max_depth']=8
    param['eta']=0.02
    param['eval_metric']='auc'
    watchlist = [ (xgb_test, 'test'),(xgb_train,'train')]
    bst = xgb.train(param, xgb_train, num_boost_round=5000, evals=watchlist)

    scores = bst.predict(xgb_dtest)
    result = pd.DataFrame({"uid":dtest_uid, "score":scores}, columns=["uid","score"])
    result.to_csv('dummy_'+str(time.time())+'.csv', index=False)


if __name__ == '__main__':
    main()
