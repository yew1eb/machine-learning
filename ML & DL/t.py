import numpy as np

X = np.array([[ 1., -1.,  2.],[ 2.,  0.,  0.],[ 0.,  1., -1.]])
print(X,'\n')
from sklearn import preprocessing
# normalize the data attributes
normalized_X = preprocessing.normalize(X)
print(normalized_X,'\n')
# standardize the data attributes
standardized_X = preprocessing.scale(X)
print(standardized_X)