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

def lowercase_word(word):
	return word.lower()

def tokenize_doc(doc):
	return word_tokenize(doc)

def create_sets(full_set, train_pct):
	#REDO to equally divide per genre
	train_set = [], test_set = []	
	train_set = set(random.sample(full_set, int(train_pct*len(full_set))))
	test_set = set(full_set) - train_set
	return (train_set, test_set)

def get_full_set():
	full_set = []
	for genre in GENRES:	
		full_set = map(lambda doc: (doc, genre), get_docs_from_genre(genre))
	return full_set

def get_docs_from_genre(genre):
	#TODO
	return []

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

def classifier_metrics(predicted_labels, correct_labels):
	#REDO if classifier dependent
	acc = accuracy(correct_labels, predicted_labels)
	prec = precision(correct_labels, predicted_labels)
	rec = recall(correct_labels, predicted_labels)
	f1 = f_measure(correct_labels, predicted_labels)
	return {accuracy: acc, precision: prec, recall: rec, f_measure: f1}