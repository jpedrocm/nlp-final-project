import random
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.classify import SklearnClassifier
from nltk.metrics import *
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

GENRES = []
STOPWORDS = stopwords.words('portuguese')

def mean(arr):
	return np.mean(arr, dtype=np.float64)

def std_dev(arr):
	return np.std(arr, dtype=np.float64)

def stem_word(word):
	return RSLPStemmer().stem(word)

def remove_stopwords_from_doc(tokenized_doc):
	return filter(lambda word: word not in STOPWORDS, tokenized_doc)

def lowercase_word(word):
	return word.lower()

def tokenize_doc(doc):
	return word_tokenize(doc)

def create_sets(full_set, train_pct):
	#TODO 
	#equally divide per genre based on train_pct
	return (train_set, test_set)

def get_full_set():
	#TODO
	return full_set

def featurize_set(given_set):
	#TODO
	return given_set

def random_forest_model(num_of_trees=10, max_depth=None, num_of_cores=1):
	return RandomForestClassifier(n_estimators=num_of_trees, max_depth=max_depth, n_jobs=num_of_cores)

def multinomial_naive_bayes_model(alpha=1.0, fit_prior=True):
	return MultinomialNB(alpha=alpha, fit_prior=fit_prior)

def logistic_regression_model(max_iter=100, multi_class='ovr', num_of_cores = 1):
	return LogisticRegression(max_iter=max_iter, multi_class=multi_class, n_jobs=num_of_cores)

def linear_svc_model(max_iter=1000):
	return LinearSVC(max_iter=max_iter)

def svc_model(kernel='rbf', degree=3, coef0=0.0, max_iter=-1, decision_function_shape='ovo'):
	return SVC(kernel=kernel, degree=degree, coef0=coef0, max_iter=max_iter, decision_function_shape=decision_function_shape)

def create_classifier(sklearn_model):
	return SklearnClassifier(sklearn_model)

def train_classifier(sklearn_classifier, train_set):
	return sklearn_classifier.train(train_set)

def test_classifier(sklearn_classifier, test_set):
	return sklearn_classifier.classify_many(test_set)

def classifier_metrics():
	#TODO
	return 0

def print_metrics(metrics):
	print

def test(stem, case_folding, no_stopwords, lowercase):
	#PRE-PROCESS
	full_set = get_full_set()
	full_preprocessed_set = preprocess_set(full_set)
	train_set, test_set = create_sets(full_preprocessed_set, 0.7)
	ready_train_set = featurize_set(train_set)
	ready_test_set = featurize_set(test_set)

	for model_string in MODELS:
		#CLASSIFICATION
		clf = create_classifier(MODELS[model_string])
		trained_clf = train_classifier(clf, ready_train_set)
		tested_clf = test_classifier(trained_clf, ready_test_set)

		#METRICS
		metrics = classifier_metrics()
		print "TEST NUMBER\n" + str(test_number)
		print 'STEMMING: ' + str(stem)
		print 'CASE-FOLDING: ' + str(case_folding)
		print 'NO-STOPWORDS: ' + str(no_stopwords)
		print 'LOWERCASE: ' + str(lowercase)
		print 'CLF = ' + model_string
		print metrics

BOOLS = [True, False]
EXPERIMENTS = [(stem, case_folding, no_stopwords, lowercase) for stem in BOOLS for case_folding in BOOLS for no_stopwords in BOOLS, for lowercase in BOOLS]
MODELS = {'NAIVE BAYES DEFAULT': multinomial_naive_bayes_model()}

def experiment():
	for e in EXPERIMENTS:
		stem = e[0], case_folding = e[1], no_stopwords = e[2], lowercase = e[3]
		test(stem, case_folding, no_stopwords, lowercase)