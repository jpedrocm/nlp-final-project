# coding: utf-8

import random, os, json
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

GENRES = [u"Axé", "Funk Carioca", "MPB", "Samba", "Sertanejo"]
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
	full_set = {}
	for genre in GENRES:
		full_set[genre] = list()

	genre_files = list()
	min_file_size = 5000

	data_path = os.getcwd() + "\lyrics_music\Data\lyrics\\"
	for filename in os.listdir(data_path):
		filepath = data_path+filename
		cur_json = json.load(open(filepath, 'r'))
		genre_files.append(cur_json)
		min_file_size = min(min_file_size, len(cur_json))

	for genre_file in genre_files:
		filtered_genre_file = random.sample(genre_file, min_file_size)
		for item in filtered_genre_file:
			document_pair = (item['title'], item['lyrics'])
			full_set[item['genre']].append(document_pair)
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

def save_genre_metrics(reference_list, test_list, genre):
	#row = ref, col = test
	non_genre = 'non_'+genre

	conf_matrix = ConfusionMatrix(reference_list, test_list)
	tp = conf_matrix[genre][genre]
	tn = conf_matrix[non_genre][non_genre]
	fp = conf_matrix[non_genre][genre]
	fn = conf_matrix[genre][non_genre]

	accuracy = float(tp+tn)/(tp+tn+fp+fn)
	precision = float(tp)/(tp+fp)
	recall = float(tp)/(tp+fn)
	f1 = float(2*precision*recall)/(precision+recall)
	
	metric = {}
	metric['tp'] = tp
	metric['tn'] = tn
	metric['fn'] = fn
	metric['fp'] = fp
	metric['accuracy'] = accuracy
	metric['precision'] = precision
	metric['recall'] = recall
	metric['f1'] = f1

	return metric

def calculate_classifier_metrics(genres_metrics):
	metrics = {}
	num_of_classes = len(genres_metrics)

	tp_sum = sum([genres_metrics[genre]['tp'] for genre in genres_metrics])
	tn_sum = sum([genres_metrics[genre]['tn'] for genre in genres_metrics])
	fp_sum = sum([genres_metrics[genre]['fp'] for genre in genres_metrics])
	fn_sum = sum([genres_metrics[genre]['fn'] for genre in genres_metrics])

	metrics['macro'] = {}
	metrics['macro']['tp'] = tp_sum/num_of_classes
	metrics['macro']['tn'] = tn_sum/num_of_classes
	metrics['macro']['fp'] = fp_sum/num_of_classes
	metrics['macro']['fn'] = fn_sum/num_of_classes
	metrics['macro']['accuracy'] = sum([genres_metrics[genre]['accuracy'] for genre in genres_metrics])/num_of_classes
	metrics['macro']['precision'] = sum([genres_metrics[genre]['precision'] for genre in genres_metrics])/num_of_classes
	metrics['macro']['recall'] = sum([genres_metrics[genre]['recall'] for genre in genres_metrics])/num_of_classes
	metrics['macro']['f1'] = sum([genres_metrics[genre]['f1'] for genre in genres_metrics])/num_of_classes

	micro_precision = tp_sum/(tp_sum+fp_sum)
	micro_recall =	tp_sum/(tp_sum+fn_sum)

	metrics['micro']= {}
	metrics['micro']['tp'] = tp_sum
	metrics['micro']['tn'] = tn_sum
	metrics['micro']['fp'] = fp_sum
	metrics['micro']['fn'] = fn_sum
	metrics['micro']['accuracy'] = 
	metrics['micro']['precision'] = micro_precision
	metrics['micro']['recall'] = micro_recall
	metrics['micro']['f1'] = 2*micro_precision*micro_recall/(micro_precision+micro_recall)

def write_case_to_file(stem, case_folding, no_stopwords, lowercase, model_name, f_type):
	global METRICS_FILE

	METRICS_FILE.write("TEST NUMBER\n" + str(TEST_NUMBER))
	METRICS_FILE.write('STEMMING: ' + str(stem))
	METRICS_FILE.write('CASE-FOLDING: ' + str(case_folding))
	METRICS_FILE.write('NO-STOPWORDS: ' + str(no_stopwords))
	METRICS_FILE.write('LOWERCASE: ' + str(lowercase))
	METRICS_FILE.write('CLF = ' + model_name)
	METRICS_FILE.write('TYPE = ' + f_type)
	METRICS_FILE.write('\n')

def write_metric_to_file(metric):
	global METRICS_FILE

	METRICS_FILE.write('TP = ' + str(metric['tp']))
	METRICS_FILE.write('TN = ' + str(metric['tn']))
	METRICS_FILE.write('FP = ' + str(metric['fp']))
	METRICS_FILE.write('FN = ' + str(metric['fn']))
	METRICS_FILE.write('ACCURACY = ' + str(metric['accuracy']))
	METRICS_FILE.write('PRECISION = ' + str(metric['precision']))
	METRICS_FILE.write('RECALL = ' + str(metric['recall']))
	METRICS_FILE.write('F-MEASURE = ' + str(metric['f1']))
	METRICS_FILE.write('\n')

def write_genre_metrics_to_file(metrics):
	global METRICS_FILE

	for (genre, metric) in metrics:
		METRICS_FILE.write('GENRE: '+ genre)
		write_metric_to_file(metric)

def write_classifier_metrics_to_file(metrics):
	global METRICS_FILE

	for (m_type, metric) in metrics:
		METRICS_FILE.write('TYPE: '+ m_type)
		write_metric_to_file(metric)

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
			genres_metrics = {}
			for genre in GENRES:
				transformed_predicted_labels = transform_to_genre_labels(predicted_labels, genre)
				transformed_correct_labels = transform_to_genre_labels(correct_labels, genre)
				genres_metrics[genre] = save_genre_metrics(transformed_correct_labels, transformed_predicted_labels, genre)

			metrics = calculate_classifier_metrics(genres_metrics)
			write_case_to_file(stem, case_folding, no_stopwords, lowercase, model_name, f_type)
			write_genre_metrics_to_file(genres_metrics)
			write_classifier_metrics_to_file(metrics)

MODELS = {'NAIVE BAYES DEFAULT': multinomial_naive_bayes_model()}

def experiment():
	for e in EXPERIMENTS:
		stem = e[0], case_folding = e[1], no_stopwords = e[2], lowercase = e[3]
		test(stem, case_folding, no_stopwords, lowercase)
	METRICS_FILE.close()