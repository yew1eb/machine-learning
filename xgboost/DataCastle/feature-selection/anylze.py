import pandas as pd
import matplotlib.pyplot as plt

path = 'd:/dataset/rp/'
features_type_csv = path + 'features_type.csv'
features = pd.read_csv(features_type_csv)
numeric = features.feature[features.type == 'numeric']
category = features.feature[features.type == 'category']

print('feature\nnumeric: %d ; category: %d' % (numeric.shape[0], category.shape[0]) )

feature_score_csv = './0.70_feature_score.csv'
feature_score = pd.read_csv(feature_score_csv)

feature_score.index = feature_score.feature
feature_score = feature_score.drop(['feature'], axis=1)

print('feature__category')
feature_score_category = feature_score.ix[category]
feature_score_category = feature_score_category.sort_values(by='fscore', ascending=False)
feature_score_category.to_csv('./feature_score_category.csv')

category_is_null = feature_score_category[feature_score_category.fscore.isnull()]
list_null = list(category_is_null.index)
print(list_null)

print('feature__numeric')
feature_score_numeric = feature_score.ix[numeric]
feature_score_numeric = feature_score_numeric.sort_values(by='fscore', ascending=False)
feature_score_numeric.to_csv('./feature_score_numeric.csv')
