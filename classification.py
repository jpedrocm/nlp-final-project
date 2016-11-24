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
BOOLS = [True, False]
EXPERIMENTS = [(stem, case_folding, no_stopwords, lowercase) for stem in BOOLS for case_folding in BOOLS for no_stopwords in BOOLS for lowercase in BOOLS]
FEATURE_TYPES = ['BINARY', 'TF', 'LOG_TF', 'TF_IDF']
TEST_NUMBER = 0
METRICS_FILE = open('metrics_file.out', 'w')

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

def preprocess_set(given_set, stem, case_folding, no_stopwords, lowercase):
	#TODO
	return given_set

def create_sets(full_set, train_pct):
	#TODO 
	#equally divide per genre based on train_pct
	return (train_set, test_set)

def get_full_set():
	#TODO 
	#get data
	return full_set

def featurize_set(given_set, feature_type):
	#TODO
	#choose a feature type and apply to docs in set
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

def transform_to_genre_labels(instances_labels, genre):
	return [label if label==genre else 'non_'+genre for label in instances_labels]

def classifier_metrics(reference_list, test_list, genre):
	#row = ref, col = test
	non_genre = 'non_'+genre

	conf_matrix = ConfusionMatrix(reference_list, test_list)
	tp = conf_matrix[genre][genre]
	tn = conf_matrix[non_genre][non_genre]
	fp = conf_matrix[non_genre][genre]
	fn = conf_matrix[genre][non_genre]
	
	metric = {}
	metric[genre] = {}
	metric[genre]['tp'] = tp
	metric[genre]['tn'] = tn
	metric[genre]['fn'] = fn
	metric[genre]['fp'] = fp
	metric[genre]['accuracy'] = accuracy
	metric[genre]['precision'] = precision
	metric[genre]['recall'] = recall
	metric[genre]['f1'] = f1

	return metric

def write_metrics_to_file(metrics, stem, case_folding, no_stopwords, lowercase, model_name, f_type):
		global METRICS_FILE

		METRICS_FILE.write("TEST NUMBER\n" + str(TEST_NUMBER))
		METRICS_FILE.write('STEMMING: ' + str(stem))
		METRICS_FILE.write('CASE-FOLDING: ' + str(case_folding))
		METRICS_FILE.write('NO-STOPWORDS: ' + str(no_stopwords))
		METRICS_FILE.write('LOWERCASE: ' + str(lowercase))
		METRICS_FILE.write('CLF = ' + model_name)
		METRICS_FILE.write('TYPE = ' + f_type)
		METRICS_FILE.write('\n')

		for (genre, metric) in metrics:
			METRICS_FILE.write('GENRE: '+ genre)
			METRICS_FILE.write('TP = ' + str(metric['tp']))
			METRICS_FILE.write('TN = ' + str(metric['tn']))
			METRICS_FILE.write('FP = ' + str(metric['fp']))
			METRICS_FILE.write('FN = ' + str(metric['fn']))
			METRICS_FILE.write('ACCURACY = ' + str(metric['accuracy']))
			METRICS_FILE.write('PRECISION = ' + str(metric['precision']))
			METRICS_FILE.write('RECALL = ' + str(metric['recall']))
			METRICS_FILE.write('F-MEASURE = ' + str(metric['f1']))
			METRICS_FILE.write('\n')

def test(stem, case_folding, no_stopwords, lowercase):
	global TEST_NUMBER

	#PRE-PROCESS
	full_set = get_full_set()
	full_preprocessed_set = preprocess_set(full_set, stem, case_folding, no_stopwords, lowercase)
	train_set, test_set = create_sets(full_preprocessed_set, 0.7)

	correct_labels = 0 #TODO get labels from test set

	for f_type in FEATURE_TYPES:
		#FEATURIZATION
		ready_train_set = featurize_set(train_set)
		ready_test_set = featurize_set(test_set)

		for model_name in MODELS:
			TEST_NUMBER+=1

			#CLASSIFICATION
			clf = create_classifier(MODELS[model_name])
			trained_clf = train_classifier(clf, ready_train_set)
			predicted_labels = test_classifier(trained_clf, ready_test_set)

			#METRICS
			metrics = []
			for genre in GENRES:
				transformed_predicted_labels = transform_to_genre_labels(predicted_labels, genre)
				transformed_correct_labels = transform_to_genre_labels(correct_labels, genre)
				metrics.append(classifier_metrics(transformed_correct_labels, transformed_predicted_labels, genre))
			write_metrics_to_file(metrics, stem, case_folding, no_stopwords, lowercase, model_name, f_type)

MODELS = {'NAIVE BAYES DEFAULT': multinomial_naive_bayes_model()}

def experiment():
	for e in EXPERIMENTS:
		stem = e[0], case_folding = e[1], no_stopwords = e[2], lowercase = e[3]
		test(stem, case_folding, no_stopwords, lowercase)