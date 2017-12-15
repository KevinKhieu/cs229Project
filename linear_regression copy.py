#####################################################################
# linear_regression.py
# ---------------------
# CS229 Project
# 
# performs linear regression on the dataset
#
# experiment 1. for unbalanced dataset
# experiment 2. for capped dataset
# experiment 3: for binarized labels
#####################################################################

import tensorflow as tf
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from matplotlib import pyplot as plt
from datetime import datetime
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from stemmer import PorterStemmer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

'''
'''

# case 1: unbalanced original dataset
#file_name = '../pickles/pickled_chains.txt'

# case 2: balanced multi-label dataset
# file_name = '../pickles/pickled_reduced_chains.txt'

# case 3: binarize
file_name = '../pickles/balanced_chains.pickle'

# balanced buckets
emails = None
with open(file_name, "rb") as fp1:   # Unpickling
    emails = pickle.load(fp1)

def GetDataset():
    np.random.shuffle(emails)

    x_vals = []
    y_vals = []

    i = 0
    text_data = []
    for i in range(0, len(emails)):
        #print "Evaluation Email %d" % (i)
        # note: time diff in mins

        # for experiment 3
        email, next_email, time_diff, label = emails[i]

        # for experiment 1, 2
        #email, next_email, time_diff = emails[i]

        # Create feature array
        features = []

        #Feature 1: Number of to
        features.append(len(email['to']))

        # Feature 2: Num words
        words = email['body'].split()
        features.append(len(words))

        # Feature 3: Number of CC
        features.append(email['cc_count'])

        # Feature 4: is reply
        if email['is_re']:
            features.append(1)
        else:
            features.append(0)

        # Feature 5: Time of Day (hour)
        date = email['date']['local_date']
        hour = date.hour
        features.append(hour)

        # Feature 6: Length of Subject Line
        subject_words = email['subject'].split()
        features.append(len(subject_words))

        # Feature 7: Day of Week
        features.append(date.weekday())

        # Feature 8: Question marks in Body
        features.append(email['body'].count('?'))

        # Feature 9: Question marksin Subject
        features.append(email['subject'].count('?'))
        
        # Append y_value for training point: hours it took to respond
        hours_to_respond = int(np.round(time_diff / 60.0))

        if hours_to_respond > 24: continue

        x_vals.append(features)
        
        #y_vals.append(hours_to_respond)
        y_vals.append(label)

    X = np.array(x_vals)
    Y = np.array(y_vals)
    print len(X), len(Y), '\n'
    return X, Y

def getAccuracy(Y_pred, Y):
    nCorrect = 0
    for i in range(len(Y)):
        y_hat = int(np.round(Y_pred[i]))
        y = Y[i]

        if y_hat == y: nCorrect += 1

    accuracy = float(nCorrect) / len(Y)
    return accuracy    

def linearRegression():
    '''
    return accuracy
    '''

    # Get the data - separate numpy arrays
    X, Y = GetDataset()

    training_cutoff = int(len(Y) * 0.8)

    # 80% of data is training
    X_train = X[:training_cutoff]
    Y_train = Y[:training_cutoff]

    # 20% of data is test
    X_test = X[training_cutoff:]
    Y_test = Y[training_cutoff:]

    
    # Create linear regression object
    #regr = linear_model.LinearRegression()
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    print len(X_train), len(X_test)
    regr.fit(X_train, Y_train)

    # Make predictions: on test set and on train set
    Y_test_pred = regr.predict(X_test)
    Y_train_pred = regr.predict(X_train)

    # get test and train accuracies
    test_accuracy = getAccuracy(Y_test_pred, Y_test)
    train_accuracy = getAccuracy(Y_train_pred, Y_train)

    print 'test accuracy:', test_accuracy 
    print 'train accuracy: ', train_accuracy
    print

    return test_accuracy, train_accuracy

def main():
    #linearRegression()

    NUM_RUNS = 10

    # run linear regression
    test_accuracies = []
    train_accuracies = []
    for i in range(NUM_RUNS):
        test_acc, train_acc = linearRegression()
        test_accuracies += [test_acc]
        train_accuracies += [train_acc]

    avg_test_accuracy = np.mean(test_accuracies)
    avg_train_accuracy = np.mean(train_accuracies)

    print 'average test accuracy:', avg_test_accuracy 
    print 'average train accuracy:', avg_train_accuracy 

if __name__ == '__main__':
    main()