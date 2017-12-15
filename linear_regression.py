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

def GetTimeDifference(date1, date2):
	return (date2-date1).seconds / 60.0

def GetDataset():
	emails = None
	x_vals = []
	y_vals = []
	stemmer = PorterStemmer()

	with open("pickled_reduced_chains.txt", "rb") as fp1:   # Unpickling
		emails = pickle.load(fp1)
	i = 0
	text_data = []
	for i in range(0, len(emails)):
		print "Evaluation Email %d" % (i)
		email, next_email, time_diff = emails[i]
		print emails[i]
		# Create feature array
		features = []
		# if np.round(time_diff / 60) > 72: continue;
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

		# Feature 5: Time of Day (minutes)
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

		x_vals.append(features)

		# Append y_value for training point
		y_vals.append(int(np.round(time_diff / 60)))

	a = np.array(x_vals)
	b = np.array(y_vals)
	return a, b

def GetTFIDF():
	emails = None
	x_vals = []
	y_vals = []
	stemmer = PorterStemmer()
	
	# Get email chains
	with open("balanced_chains.pickle", "rb") as fp1:   # Unpickling
		emails = pickle.load(fp1)

	np.random.shuffle(emails)
	i = 0
	text_data = []
	for i in range(0, len(emails)):
		print "Evaluation Email %d" % (i)
		email, next_email, time_diff, bucket = emails[i]
		# if int(np.round(time_diff / 60)) > 72:
		# 	continue
		# Create stemmed body and append to text_data
		new_str = ""
		words = email['body'].split()
		for word in words:
			new_word = stemmer.stem(word, 0, len(word) - 1)
			new_str += new_word + " "
		new_str = new_str[:-1]
		text_data.append(new_str)

		# Append hour
		y_vals.append(int(np.round(time_diff / 60)))
		#y_vals.append(int(time_diff)

	b = np.array(y_vals)
	count_vect = CountVectorizer()
	X_train_counts = count_vect.fit_transform(text_data)
	tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
	X_train_tf = tf_transformer.transform(X_train_counts)
	return X_train_tf, b, count_vect, tf_transformer, text_data


def BagOfWords():
	total_features, total_targets, count_vect, tfidf_transformer, body = GetTFIDF()

	# Keep 80% samples for training
	training_cutoff = int(len(total_targets) * .9)
	train_features = total_features[:training_cutoff]
	train_targets = total_targets[:training_cutoff]

	# Keep remaining samples as test set
	test_features = body[training_cutoff:]
	test_targets = total_targets[training_cutoff:]

	# clf
	clf = MultinomialNB().fit(train_features, train_targets)

	# Create tfidf matrices with test features
	X_new_counts = count_vect.transform(test_features)
	X_new_tfidf = tfidf_transformer.transform(X_new_counts)


	#predicted = clf.predict(X_new_tfidf)
	predicted = clf.predict(train_features)
	num_correct = 0
	num_total = 0
	for i in range(1, len(predicted) + 1):
		print('Sample %d Prediction => %d' % (i, predicted[i - 1]))
		print('Sample %d Real => %d' % (i, train_targets[i - 1]))
		if predicted[i - 1] == train_targets[i - 1]:
			num_correct += 1
		num_total += 1
		print "-----------------------\n"
	print "Final Num Correct: %d" % (num_correct)
	print "Final Total: %d" % (num_total)
	print "Final Percent: %1.4f" % (float(num_correct) / float(num_total))




def LinearRegression():
	# Get the data - separate numpy arrays
	total_features, total_targets = GetDataset()

	training_cutoff = int(len(total_targets) * .7)
	validation_cutoff = int(len(total_targets) * .8)

	# 70% of data is training
	train_features = scale(total_features[:training_cutoff])
	train_targets = total_targets[:training_cutoff]

	# 10% of data is validation
	valid_features = scale(total_features[training_cutoff:validation_cutoff])
	valid_targets = total_targets[training_cutoff:validation_cutoff]

	# 20% of data is test
	test_features = scale(total_features[validation_cutoff:])
	test_targets = total_targets[validation_cutoff:]

	# Create Weight (W) and Offset (b) matrices
	num_features = len(total_features[0])
	w = tf.Variable(tf.truncated_normal([num_features, 1], mean=0.0, stddev=1.0, dtype=tf.float64))
	b = tf.Variable(tf.zeros(1, dtype = tf.float64))
	x = tf.placeholder(tf.int32, [1])
	y = tf.placeholder(tf.int32, [1])
	# print len(test_features[0])
	# print len(test_targets[0])
	acc, acc_op = tf.metrics.accuracy(labels=x, predictions=y)

	# Function for calculating cost between feature and targets
	def calc(x, y_vals, sess):
		# Returns predictions and error
		predictions = tf.add(b, tf.matmul(x, w))
		error = tf.reduce_mean(tf.sqrt(tf.square(y_vals - predictions)))
		print predictions
		return [ predictions, error ]

	# Initialize training predictions and error
	y, cost = calc(train_features, train_targets, None)

	# Parameters
	learning_rate = 0.025
	epochs = 1000
	points = [[], []] # For Plotting

	# Initialize tensor global variables and create gradient descent optimizer
	init = tf.global_variables_initializer()
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
	num_correct = 0
	num_total = 0
	with tf.Session() as sess:

	    sess.run(init)

	    for i in list(range(epochs)):
	    	print "Epoch %d" % (i)
	        sess.run(optimizer)

	        if i % 10 == 0.:
	            points[0].append(i+1)
	            points[1].append(sess.run(cost))

	        if i % 100 == 0:
	            print(sess.run(cost))

		valid_target = calc(valid_features, valid_targets, sess)[1]

		print('Validation error =', sess.run(valid_target), '\n')

		test_target = calc(test_features, test_targets, sess)[1]

		print('Test error =', sess.run(test_target), '\n')



	# plt.plot(points[0], points[1], 'r--')
	# plt.axis([0, epochs, 50, 600])
	# plt.show()



BagOfWords()
#LinearRegression()