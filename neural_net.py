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
from sklearn.neural_network import MLPClassifier
import math
import re
import random

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

'''

'''

def GetFeatures(emails):

	#np.random.shuffle(emails)
	word_mapping = GetWordDictionary(emails)

	# with open("balanced_chains_shuffed.pickle", "wb") as fp2:
	# 	pickle.dump(emails, fp2, protocol=pickle.HIGHEST_PROTOCOL)
	stemmer = PorterStemmer()
	training_cutoff = int(len(emails) * 0.9)
	x_vals = []
	y_vals = []
	count_0 = 0
	count_1 = 0
	for i in range(0, len(emails)):
		#print "Evaluation Email %d" % (i)
		email, next_email, time_diff, bucket = emails[i] #, bucket

		if (float(np.round(time_diff / 60)) > 24):
			continue
		num_features = 9 #15 #13
		num_words =  0 #len(word_mapping)
		# Create feature array
		features = np.zeros(shape=(num_features + num_words))

		#Feature 1: Number of to
		features[0] = float(len(email['to']))


		# Feature 2: Num words
		words = email['body'].split()
		lower_case_body = [stemmer.stem(x.lower(), 0, len(x) - 1) for x in words]
		features[1] = float(len(words))

		# print email
		# print bucket
		# print lower_case_body
		# print "-------------\n\n"

		#Feature 3: Number of CC
		# Old: 0.5442, 0.5387
		features[2] = float(email['cc_count'])

		# Feature 4: is reply
		# without, .6298, with. .5994
		if email['is_re']:
			features[3] = 1.0
		else:
			features[3] = 0.0

		# Feature 5: Time of Day (hour)
		#print email
		date = email['date']['local_date']
		hour = date.hour
		# Old, 0.5442, New = 0.5276
		features[4] = float(hour)

		#Feature 6: Length of Subject Line
		subject_words = email['subject'].split()
		lower_case_words = [stemmer.stem(x.lower(), 0, len(x) - 1) for x in subject_words]
		features[5] = int(len(subject_words))

		# Feature 7: Day of Week
		# without, 0.5580
		features[6] = (date.weekday())

		# Feature 8: Question marks in Body
		features[7] = (email['body'].count('?'))

		# Feature 9: Question marksin Subject
		features[8] = (email['subject'].count('?'))

		# # Feature 10: "RESPONSE NEEDED" in subject
		# stemmed_response = stemmer.stem("response", 0, len("response") - 1)
		# if stemmed_response in lower_case_words:
		# 	features[9] = 1
		# else:
		# 	features[9] = 0

		# # Feature 11: "Please" in words
		# stemmed_please = stemmer.stem("please", 0, len("please") - 1)
		# if stemmed_please in lower_case_words:
		# 	features[10] = 1
		# else:
		# 	features[10] = 0

		# if email['body'].find('?') != -1:
		# 	features[11] = 1
		# else:
		# 	features[11] = 0

		# if email['subject'].find('?') != -1:
		# 	features[12] = 1
		# else:
		# 	features[12] = 0

		# stemmed_can = stemmer.stem("can", 0, len("can") - 1)
		# if stemmed_can in lower_case_words:
		# 	features[13] = 1
		# else:
		# 	features[13] = 0

		# stemmed_important = stemmer.stem("important", 0, len("important") - 1)
		# if stemmed_important in lower_case_words:
		# 	features[14] = 1
		# else:
		# 	features[14] = 0

		# if stemmed_important in lower_case_body:
		# 	features[15] = 1
		# else:
		# 	features[15] = 0

		# stemmed_urgent = stemmer.stem("urgent", 0, len("urgent") - 1)
		# if stemmed_urgent in lower_case_words:
		# 	features[16] = 1
		# else:
		# 	features[16] = 0

		# if stemmed_urgent in lower_case_body:
		# 	features[17] = 1
		# else:
		# 	features[17] = 0

		# stemmed_need = stemmer.stem("need", 0, len("need") - 1)
		# if stemmed_need in lower_case_words:
		# 	features[18] = 1
		# else:
		# 	features[18] = 0

		# if stemmed_need in lower_case_body:
		# 	features[19] = 1
		# else:
		# 	features[19] = 0



		# body = SplitText(email['body'])
		# for word in body:
		# 	modified_word = word
		# 	if len(modified_word) > 1:
		# 		modified_word = stemmer.stem(word, 0, len(word) - 1).lower()

		# 	if modified_word in word_mapping:
		# 		features[word_mapping[modified_word]] = 1

		x_vals.append(features)
		y_vals.append(bucket)
		#y_vals.append(float(np.round(time_diff / 60)))
		#Append y_value for training point
		# if float(time_diff) > 60.0:
		# 	y_vals.append(1)
		# 	count_1 += 1
		# else:
		# 	y_vals.append(0)
		# 	count_0 += 1
		#y_vals.append(float(np.round(time_diff / 60)))

	#print count_0
	#print count_1
	return x_vals, y_vals

# Stack Overflow: https://stackoverflow.com/questions/9457832/python-list-rotation
def rotate(l, n):
    return l[n:] + l[:n]

# Cross Entropy Loss Reduction
def cross_entropy(true_label, prediction):
	if true_label == 1:
		return -math.log10(prediction)
	else:
		return -math.log10(1 - prediction)

def NeuralNet(clf):
	#total_features, total_targets, count_vect, tfidf_transformer, body = GetTFIDF()
	emails = None
	# with open("balanced_chains_shuffed.pickle", "rb") as fp1:   # Unpickling
	# 	emails = pickle.load(fp1)

	with open("balanced_chains.pickle", "rb") as fp1:   # Unpickling
		emails = pickle.load(fp1)

	accuracies = []
	accuracy_total = 0.0
	train_accuracy_total = 0.0
	maxVal = None
	for i in range(0, 10):
		new_emails = emails
		np.random.shuffle(new_emails)
		X, Y = GetFeatures(new_emails)
		#print "iteration %d" % (i)
		# hidden layer (15) : 
		# hidden layer (30, 5): .5856
		# (60, 30): .5663

		tenth = int(len(Y) / 10)
		for i in range(0, 1):
			test_features = X[:tenth]
			test_targets = Y[:tenth]

			train_features = X[tenth:]
			train_targets = Y[tenth:]
			clf.fit(train_features, train_targets)
			total_cost = 0.0
			num_correct = 0
			num_total = 0
			count_0 = 0
			count_1 = 0
			train_num_correct = 0
			train_num_total = 0
			# for k in range(0, len(train_targets)):
			# 	prediction = clf.predict(train_features[k])
			# 	actual = train_targets[k]
			# 	# print actual
			# 	# print "Iteration %d Sample %d=> \nPredicted: %d" % (i, j, prediction)
			# 	# print "Actual: %d" % (actual)
			# 	if prediction == actual:
			# 		train_num_correct += 1

			# 	# if actual == 1:
			# 	# 	count_1 += 1
			# 	# else:
			# 	# 	count_0 += 1
			# 	train_num_total += 1

			for j in range(0, len(test_targets)):
				prediction = clf.predict(test_features[j])
				actual = test_targets[j]
				# print actual
				# print "Iteration %d Sample %d=> \nPredicted: %d" % (i, j, prediction)
				# print "Actual: %d" % (actual)
				if prediction == actual:
					num_correct += 1

				# if actual == 1:
				# 	count_1 += 1
				# else:
				# 	count_0 += 1
				num_total += 1

			#print "Final Num Correct: %d" % (num_correct)
			#print "Final Total: %d" % (num_total)
			#print "Final Percent: %1.4f" % (float(num_correct) / float(num_total))
			#print count_0
			#print count_1
			# X = rotate(X, tenth)
			# Y = rotate(Y, tenth)

			#print "FINAL COST: %1.4f" % (total_cost)
			#accuracies.append((float(num_correct) / float(num_total)))
			accuracy_total += ((float(num_correct) / float(num_total)))
			#train_accuracy_total += ((float(train_num_correct)) / float(train_num_total))
			#if maxVal is None or ((float(num_correct) / float(num_total))) > maxVal:
			#	maxVal = (float(num_correct) / float(num_total))

	#print "FINAL ACCURACY: %1.4f" % (accuracy_total / float(10))
	#print "MAX ACCURACY: %1.4f" % (maxVal)

	#print "Final Train Accuracy: %1.4f" % (train_accuracy_total / 10.0)
	val = float(accuracy_total) / 10.0
	print "FINAL ACCURACY: %1.4f" % (val)
	return val


fh = open("results.txt","w")


clf1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(60, 60, 60, 60, 60, 60), random_state=1)
accuracy_1 = NeuralNet(clf1)
fh.write("Hidden Layers: (6), Hidden Layer Neurons: (75), Learning Rate: (1e-5)\n")
str1 = "Final Accuracy: %1.4f\n" % (accuracy_1)
fh.write(str1)

clf2 = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(60, 60, 60, 60, 60, 60), random_state=1)
accuracy_2 = NeuralNet(clf2)
fh.write("Hidden Layers: (6), Hidden Layer Neurons: (90), Learning Rate: (1e-4)\n")
str2 = "Final Accuracy: %1.4f\n" % (accuracy_2)
fh.write(str2)

clf3 = MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(60, 60, 60, 60, 60, 60), random_state=1)
accuracy_3 = NeuralNet(clf3)
fh.write("Hidden Layers: (6), Hidden Layer Neurons: (60), Learning Rate: (1e-3)\n")
str3 = "Final Accuracy: %1.4f\n" % (accuracy_3)
fh.write(str3)

clf4 = MLPClassifier(solver='lbfgs', alpha=1e-2, hidden_layer_sizes=(60, 60, 60, 60, 60, 60), random_state=1)
accuracy_4 = NeuralNet(clf4)
fh.write("Hidden Layers: (6), Hidden Layer Neurons: (20), Learning Rate: (1e-2)\n")
str4 = "Final Accuracy: %1.4f\n" % (accuracy_4)
fh.write(str4)

clf5 = MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(60, 60, 60, 60, 60, 60), random_state=1)
accuracy_5 = NeuralNet(clf5)
fh.write("Hidden Layers: (6), Hidden Layer Neurons: (40), Learning Rate: (1e-1)\n")
str5 = "Final Accuracy: %1.4f\n" % (accuracy_5)
fh.write(str5)

clf6 = MLPClassifier(solver='lbfgs', alpha=1e-7, hidden_layer_sizes=(60, 60, 60, 60, 60, 60), random_state=1)
accuracy_6 = NeuralNet(clf6)
fh.write("Hidden Layers: (6), Hidden Layer Neurons: (60), Learning Rate: (1e-7)\n")
str6 = "Final Accuracy: %1.4f\n\n\n" % (accuracy_6)
fh.write(str6)

clf1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(60, 60, 60, 60, 60, 60, 60), random_state=1)
accuracy_1 = NeuralNet(clf1)
fh.write("Hidden Layers: (7), Hidden Layer Neurons: (75), Learning Rate: (1e-5)\n")
str1 = "Final Accuracy: %1.4f\n" % (accuracy_1)
fh.write(str1)

clf2 = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(60, 60, 60, 60, 60, 60, 60), random_state=1)
accuracy_2 = NeuralNet(clf2)
fh.write("Hidden Layers: (7), Hidden Layer Neurons: (90), Learning Rate: (1e-4)\n")
str2 = "Final Accuracy: %1.4f\n" % (accuracy_2)
fh.write(str2)

clf3 = MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(60, 60, 60, 60, 60, 60, 60), random_state=1)
accuracy_3 = NeuralNet(clf3)
fh.write("Hidden Layers: (7), Hidden Layer Neurons: (60), Learning Rate: (1e-3)\n")
str3 = "Final Accuracy: %1.4f\n" % (accuracy_3)
fh.write(str3)

clf4 = MLPClassifier(solver='lbfgs', alpha=1e-2, hidden_layer_sizes=(60, 60, 60, 60, 60, 60, 60), random_state=1)
accuracy_4 = NeuralNet(clf4)
fh.write("Hidden Layers: (7), Hidden Layer Neurons: (20), Learning Rate: (1e-2)\n")
str4 = "Final Accuracy: %1.4f\n" % (accuracy_4)
fh.write(str4)

clf5 = MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(60, 60, 60, 60, 60, 60, 60), random_state=1)
accuracy_5 = NeuralNet(clf5)
fh.write("Hidden Layers: (7), Hidden Layer Neurons: (40), Learning Rate: (1e-1)\n")
str5 = "Final Accuracy: %1.4f\n" % (accuracy_5)
fh.write(str5)

clf6 = MLPClassifier(solver='lbfgs', alpha=1e-7, hidden_layer_sizes=(60, 60, 60, 60, 60, 60, 60), random_state=1)
accuracy_6 = NeuralNet(clf6)
fh.write("Hidden Layers: (7), Hidden Layer Neurons: (60), Learning Rate: (1e-7)\n")
str6 = "Final Accuracy: %1.4f\n" % (accuracy_6)
fh.write(str6)

# clf7 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(60, 60), random_state=1)
# accuracy_7 = NeuralNet(clf7)
# fh.write("Hidden Layers: (2), Hidden Layer Neurons: (60, 60), Learning Rate: (1e-5)\n")
# str7 = "Final Accuracy: %1.4f\n" % (accuracy_7)
# fh.write(str7)

# clf8 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(60, 30), random_state=1)
# accuracy_8 = NeuralNet(clf8)
# fh.write("Hidden Layers: (2), Hidden Layer Neurons: (60, 30), Learning Rate: (1e-5)\n")
# str8 = "Final Accuracy: %1.4f\n" % (accuracy_8)
# fh.write(str8)

# clf9 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(60, 80), random_state=1)
# accuracy_9 = NeuralNet(clf9)
# fh.write("Hidden Layers: (2), Hidden Layer Neurons: (60, 80), Learning Rate: (1e-5)\n")
# str9 = "Final Accuracy: %1.4f\n" % (accuracy_9)
# fh.write(str9)

# clf10 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(60, 30, 15), random_state=1)
# accuracy_10 = NeuralNet(clf10)
# fh.write("Hidden Layers: (3), Hidden Layer Neurons: (60, 30, 15), Learning Rate: (1e-5)\n")
# str10 = "Final Accuracy: %1.4f\n" % (accuracy_10)
# fh.write(str10)

# clf11 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(60, 30, 15, 15), random_state=1)
# accuracy_11 = NeuralNet(clf11)
# fh.write("Hidden Layers: (4), Hidden Layer Neurons: (60, 30, 15, 15), Learning Rate: (1e-5)\n")
# str11 = "Final Accuracy: %1.4f\n" % (accuracy_10)
# fh.write(str11)

fh.close()
#BagOfWords()
#LinearRegression()