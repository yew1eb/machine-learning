import xgboost as xgb
import pandas as pd
import time

def get_data():
    dtrain = pd.read_csv('dtrain.csv').iloc[:, 1:]
    dtest  = pd.read_csv('dtest.csv').iloc[:, 1:]
    labels = pd.read_csv('train_y.csv')
    return dtrain, labels, dtest

def get_binary_data():
    xgb_dtrain = xgb.DMatrix('xgb.dtrain')
    xgb_dtest  = xgb.DMatrix('xgb.dtest')
    test_uid = pd.read_csv('test_x.csv').iloc[:,0]
    return xgb_dtrain, xgb_dtest, test_uid

def main():
    #dtrain, labels, dtest = get_data()
    #xgb_dtrain = xgb.DMatrix(dtrain, label=1-labels.y)
    #xgb_dtest  = xgb.DMatrix(dtest)

    xgb_dtrain, xgb_dtest, test_uid = get_binary_data()

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
    num_round = 1820
    bst = xgb.train(param, xgb_dtrain, num_round, watchlist)

    pred = bst.predict(xgb_dtest)
    result = pd.DataFrame(test_uid)
    result['score'] = pd.Series(pred)
    result.to_csv(str(time.time())+'.csv', index=False)
if __name__ == '__main__':
    main()
