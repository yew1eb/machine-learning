import numpy as np
import pandas as pd

def get_sample(src, dest, n):
    data = pd.read_csv(src)
    sample_data = data.loc[0:n,['uid','x1','x2','x3','x4','x5','x1134','x1135','x1136','x1137','x1138']]
    sample_data.to_csv(dest)

def get_sample_label(src, dest, n):
    data = pd.read_csv(src)
    sample_data = data.loc[0:n, :]
    sample_data.to_csv(dest)

def get_small_data():
    features = pd.read_csv('features_type.csv')


if __name__ == '__main__':
    get_sample('test_x.csv', 'stest_x.csv', 5)
    get_sample('train_unlabeled.csv', 'strain_unlabeled.csv', 5)
    get_sample('train_x.csv', 'strain_x.csv', 5)
    get_sample_label('train_y.csv', 'strain_y.csv', 5)
