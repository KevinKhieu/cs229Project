#####################################################################
# logistic_regression.py
# ----------------------
# CS229 Project
# 
# performs logistic regression on the multiclass and binarized
# dataset (experiments 1-4)
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

# experiment 1: unbalanced original dataset
#file_name = '../pickles/pickled_chains.txt'

# experiment 2: balanced multi-label dataset
#file_name = '../pickles/pickled_reduced_chains.txt'

# experiment 3 & 4: binarize
file_name = '../pickles/balanced_chains.pickle'

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
        
        # for experiment 3 & 4
        email, next_email, time_diff, label = emails[i]

        # for experiment 1 & 2
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
        features.append(1 if email['is_re'] else 0)

        # Feature 5: Time of Day (hour)
        date = email['date']['local_date']
        hour = date.hour
        hot_vec = [0] * 24
        hot_vec[hour] = 1
        features += hot_vec

        # Feature 6: Length of Subject Line
        subject_words = email['subject'].split()
        features.append(len(subject_words))

        # Feature 7: Day of Week
        weekday = date.weekday()
        hot_vec = [0] * 7
        hot_vec[weekday] = 1
        features += hot_vec

        
        # Only for experiment 4
        # ---------------------
        # boolean: presence of ? in body / header
        features.append(1 if '?' in email['body'] else 0)
        features.append(1 if '?' in email['subject'] else 0)
       
        keywords = ['response', 'please', 'can', 'urgent', 'important', 'need']
        for keyword in keywords:
            features.append(1 if keyword in subject_words else 0)
            features.append(1 if keyword in words else 0)
        

        # Append y_value for training point: hours it took to respond or label
        
        # experiment 1 & 2
        hours_to_respond = int(np.round(time_diff / 60.0))
        if hours_to_respond > 24: continue
        #y_vals.append(hours_to_respond)

        # experiment 3, 4
        y_vals.append(label)

        x_vals.append(features)

    X = np.array(x_vals)
    Y = np.array(y_vals)
    print len(X), len(Y), '\n'
    return X, Y

def logisticRegression():
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

    print 'test size', len(X_test)
    print 'train size', len(X_train)

    
    # Create logistic regression object
    regr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg')

    # Train the model using the training sets
    print len(X_train), len(X_test)
    regr.fit(X_train, Y_train)

    # Make predictions: on test set and on train set
    Y_test_pred = regr.predict(X_test)
    Y_train_pred = regr.predict(X_train)

    # get test and train accuracies
    test_accuracy = regr.score(X_test, Y_test)
    train_accuracy = regr.score(X_train, Y_train)

    print 'test accuracy:', test_accuracy 
    print 'train accuracy: ', train_accuracy
    print

    return test_accuracy, train_accuracy


def main():

    NUM_RUNS = 10

    # run linear regression
    test_accuracies = []
    train_accuracies = []
    for i in range(NUM_RUNS):
        test_acc, train_acc = logisticRegression()
        test_accuracies += [test_acc]
        train_accuracies += [train_acc]

    avg_test_accuracy = np.mean(test_accuracies)
    avg_train_accuracy = np.mean(train_accuracies)

    print 'average test accuracy:', avg_test_accuracy 
    print 'average train accuracy:', avg_train_accuracy

if __name__ == '__main__':
    main()