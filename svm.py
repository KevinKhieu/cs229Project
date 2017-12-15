#####################################################################
# logistic.py
# ---------------------
# CS229 Project
# 
# performs SVM on the balanced binary label dataset
# experiment 3 & 4
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
from stemmer import PorterStemmer 
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import math
import re
import random

'''
'''

file_name = '../pickles/balanced_chains.pickle'

# balanced buckets
emails = None
with open(file_name, "rb") as fp1:   # Unpickling
    emails = pickle.load(fp1)


def SplitText(text):
    return re.findall(r"[\w']+|[.,!?;]", text)

# Creates word -> id mapping. Ids = 0 - size(Vocabulary)
def GetWordDictionary(emails):
    word_dict = {}
    count = 0
    stemmer = PorterStemmer()
    for email_case in emails:
        email = email_case[0]
        body = SplitText(email['body'])
        for word in body:
            modified_word = word
            if len(modified_word) > 1:
                modified_word = stemmer.stem(word, 0, len(word) - 1)

            if modified_word not in word_dict:
                word_dict[modified_word] = count
                count += 1
    return word_dict


def GetDataset():
    np.random.shuffle(emails)

    x_vals = []
    y_vals = []

    stemmer = PorterStemmer()
    word_mapping = GetWordDictionary(emails)

    i = 0
    text_data = []
    for i in range(0, len(emails)):
        #print "Evaluation Email %d" % (i)
        # note: time diff in mins
        email, next_email, time_diff, label = emails[i]

        # Create feature array
        features = []

        #Feature 1: Number of to
        features.append(len(email['to']))

        # Feature 2: Num words
        words = email['body'].split()
        lower_case_body = [stemmer.stem(x.lower(), 0, len(x) - 1) for x in words]
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
        lower_case_subject = [stemmer.stem(x.lower(), 0, len(x) - 1) for x in subject_words]
        features.append(len(subject_words))

        # Feature 7: Day of Week
        features.append(date.weekday())

        # Feature 8: # Question marks in Body, bool in body
        features.append(email['body'].count('?'))

        # Feature 9: # Question marks in Subject, bool in subject
        features.append(email['subject'].count('?'))
        

        # NEW FEATURES: only for experiment 4
        # -----------------------------------
        # boolean: presence of ? in body / header
        features.append(1 if '?' in email['body'] else 0)
        features.append(1 if '?' in email['subject'] else 0)

        # Feature 12-13: "RESPONSE NEEDED" in subject or body
        keywords = ['response', 'please', 'can', 'urgent', 'important', 'need']
        for keyword in keywords:
            stemmed_keyword = stemmer.stem(keyword, 0, len(keyword) - 1)
            features.append(1 if stemmed_keyword in lower_case_subject else 0)
            features.append(1 if stemmed_keyword in lower_case_body else 0)


        x_vals.append(features)
        y_vals.append(label)

    X = np.array(x_vals)
    Y = np.array(y_vals)
    return X, Y

def getAccuracy(Y_pred, Y):
    nCorrect = 0
    for i in range(len(Y)):
        y_hat = Y_pred[i]
        y = Y[i]
        if y_hat == y: nCorrect += 1

    accuracy = float(nCorrect) / len(Y)
    return accuracy  

def svm():
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

    
    # Create svc
    clf = SVC() # rbf

    # Train the model using the training sets
    clf.fit(X_train, Y_train)

    # Make predictions: on test set and on train set
    Y_test_pred = clf.predict(X_test)
    Y_train_pred = clf.predict(X_train)

    # get test and train accuracies
    test_accuracy = getAccuracy(Y_test_pred, Y_test)
    train_accuracy = getAccuracy(Y_train_pred, Y_train)

    print 'test accuracy:', test_accuracy 
    print 'train accuracy: ', train_accuracy
    print

    return test_accuracy, train_accuracy

def main():

    NUM_RUNS = 10

    test_accuracies = []
    train_accuracies = []
    for i in range(NUM_RUNS):
        test_acc, train_acc = svm()
        test_accuracies += [test_acc]
        train_accuracies += [train_acc]

    avg_test_accuracy = np.mean(test_accuracies)
    avg_train_accuracy = np.mean(train_accuracies)

    print 'average test accuracy:', avg_test_accuracy 
    print 'average train accuracy:', avg_train_accuracy 

if __name__ == '__main__':
    main()