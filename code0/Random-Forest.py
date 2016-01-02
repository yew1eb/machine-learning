import math
import numpy as np 
import random
import sklearn
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
import scipy.io as sio
from collections import Counter
import csv
from sklearn.feature_extraction import DictVectorizer





# Helper function to calculate the impurity given a particular split
def calculate_impurity(left_histogram, right_histogram):
    A = Counter(left_histogram)
    B = Counter(right_histogram)
    total_count = A + B
    total = dict(A + B)
    impurity_left = 0
    impurity_right = 0
    impurity_total = 0
    #print 'total' + str(total)
    #print 'left' + str(left_histogram)
    #print 'right' + str(right_histogram)

    for elem3 in total.keys():
        fraction_total = total[elem3] / float(sum(total.values()))
        impurity_total += fraction_total * math.log(fraction_total, 2)
    impurity_total *= -1 

    # In the case where we only have one category

    for elem in left_histogram.keys():
        fraction_left = left_histogram[elem] / float(sum(left_histogram.values()))
        impurity_left += fraction_left * math.log(fraction_left, 2)
    impurity_left *= -1

    for elem2 in right_histogram.keys():
        fraction_right = right_histogram[elem2] / float(sum(right_histogram.values()))
        impurity_right += fraction_right * math.log(fraction_right, 2)
    impurity_right *= -1


    return impurity_total - ((impurity_left + impurity_right) / 2)





class DecisionTree():
    def __init__(self, left=None, right=None, label=None):
        self.left = left 
        self.right = right 
        self.label = label
        self.cutoff = 0.00001
        self.depth = 0
        self.depth_limit = 40

    def impurity(self, left_label_hist, right_label_hist):
        return calculate_impurity(left_label_hist, right_label_hist) 

    def segmentor(self, data, labels): 
        min_impurity = float('inf')
        feature_index = None
        threshold = None
        data = np.array(data)
        num_columns = data.shape[1] 
        final_left_data = []
        final_right_data = []
        final_left_labels = []
        final_right_labels = []
        for index in range(num_columns): 
            left_data = {}
            right_data = {}
            feature_data = data[:,index]
            #print np.unique(feature_data)
            median = np.mean(feature_data)
            #print median
            #print "************"
            left_samples = []
            right_samples = []
            left_labels = []
            right_labels = []
            # Get the left and right histograms that result by splitting on a particular feature
            for iterator in range(len(data)):
                elem = data[iterator]
                if elem[index] < median:
                    left_samples.append(elem)
                    left_labels.append(labels[iterator])
                    if labels[iterator] in left_data.keys():
                        left_data[labels[iterator]] += 1
                    else:
                        left_data[labels[iterator]] = 1
                else:
                    right_labels.append(labels[iterator])
                    right_samples.append(elem)
                    if labels[iterator] in right_data.keys():
                        right_data[labels[iterator]] += 1
                    else:
                        right_data[labels[iterator]] = 1
            #print "Impurity with split" + str(index) 
            #print "Median is" + str(median)
            feature_impurity = self.impurity(left_data, right_data)
            # Check based on impurity to determine which feature to split on

            if feature_impurity < min_impurity:
                min_impurity = feature_impurity
                feature_index = index
                threshold = median 
                final_left_data = left_samples
                final_right_data = right_samples
                final_left_labels = left_labels
                final_right_labels = right_labels
        return (feature_index, threshold, min_impurity, final_left_data, final_left_labels, final_right_data, final_right_labels)
    def train(self, data, labels):
        num_unique = np.unique(labels)
        if len(num_unique) == 1:
            # Picking an arbitrary label, since all labels are the same it doesn't matter which one I pick
            self.label = labels[0]
        else:
            segment = self.segmentor(data, labels)
            self.split_rule = (segment[0], segment[1])
            min_impurity = segment[2]
            left_data = segment[3]
            left_labels = segment[4]
            right_data = segment[5]
            right_labels = segment[6] 
            # Check to see if there is little gain in information, in which case I stop to prevent overfitting
            if min_impurity < self.cutoff:
                most_frequent_left = np.argmax(np.bincount(left_labels))
                most_frequent_right = np.argmax(np.bincount(right_labels))
                self.left = DecisionTree()
                self.left.depth = self.depth + 1
                self.right = DecisionTree()
                self.right.depth = self.depth + 1
                self.left.label = most_frequent_left 
                self.right.label = most_frequent_right
            else:
                if len(np.unique(left_data)) > 1:
                    self.left = DecisionTree()
                    self.left.depth = self.depth + 1
                    if self.left.depth < self.depth_limit:
                        self.left.train(left_data, left_labels)
                    elif left_labels:
                        most_freqent_left = np.argmax(np.bincount(left_labels))
                        self.left.label = most_freqent_left

                elif left_labels:
                    most_freqent_left = np.argmax(np.bincount(left_labels))
                    self.left = DecisionTree()
                    self.left.depth = self.depth + 1
                    self.left.label = most_freqent_left
                if len(np.unique(right_data)) > 1:
                    self.right = DecisionTree()
                    self.right.depth = self.depth + 1
                    if self.right.depth < self.depth_limit:
                        self.right.train(right_data, right_labels)
                    elif right_labels:
                        most_frequent_right = np.argmax(np.bincount(right_labels))
                        self.right.label = most_frequent_right
                elif right_labels:
                    most_frequent_right = np.argmax(np.bincount(right_labels))
                    self.right = DecisionTree()
                    self.right.label = most_frequent_right
    def predict_one(self, data):
        # In the case where we reach a leaf
        if self.label != None:
            return self.label 
        else:
            feature_index = self.split_rule[0]
            threshold = self.split_rule[1]
            particular_value = data[feature_index]
            if particular_value < threshold:
                print self.left                                
                return self.left.predict_one(data)
            else:
                return self.right.predict_one(data)
    def predict(self, data):
        predictions = []
        for elem in data:
            predictions.append(self.predict_one(elem))
        return predictions

    def score(self, data, labels):
        num_correct = 0
        for index in range(len(data)):
            elem = data[index]
            prediction = self.predict_one(elem)
            if prediction == labels[index]:
                num_correct += 1
        return num_correct / float(len(data))



class RandomForest():
    def __init__(self, num_estimators):
        self.num_estimators = num_estimators
        self.trees = []
        self.oob_score = 0
    def train(self, data, labels):
        total_features = data.shape[1]
        num_features = math.floor(math.sqrt(total_features))
        for index in range(self.num_estimators):
            particular_features = random.sample(range(total_features), int(num_features))
            data_list = []
            for feature in particular_features:
                data_list.append(data[:,feature])
            tup = tuple(data_list)
            new_data = np.vstack(tup).T


            tree = DecisionTree()
            num_points = len(new_data)
            indices = range(len(new_data))
            num_samples = math.floor(num_points * 0.66)
            num_tests = num_points - num_samples 
            sample_indices = random.sample(indices, int(num_samples))
            test_indices = [elem for elem in indices if elem not in sample_indices]
            train_data = []
            train_labels = []
            validate_data = []
            validate_labels = []
            for elem in sample_indices:
                train_data.append(new_data[elem])
                train_labels.append(labels[elem])
            for elem2 in test_indices:
                validate_data.append(data[elem])
                validate_labels.append(labels[elem])
            tree.train(np.array(train_data), np.array(train_labels))
            self.oob_score += tree.score(np.array(validate_data), np.array(validate_labels))
            self.trees.append(tree)
        self.oob_score /= float(self.num_estimators)

    def predict_one(self, data):
        votes = []
        for tree in self.trees:
            votes.append(tree.predict_one(data))
        return np.argmax(np.bincount(np.array(votes)))


    def predict(self, data):
        predictions = []
        for elem in data:   
            predictions.append(self.predict_one(elem))

    def score(self, data, labels):
        num_correct = 0
        for index in range(len(data)):
            elem = data[index]
            prediction = self.predict_one(elem)
            if prediction == labels[index]:
                num_correct += 1
        return num_correct / float(len(data))





