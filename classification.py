# coding: utf-8

import random, os, json, time
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
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

M_FILENAME = 'metrics_file_tunning_done_no_punct.json'
GENRES = ["Sertanejo", "Funk Carioca", u"Axé",  "MPB", "Samba"]
PT_STOPWORDS = stopwords.words('portuguese')
PT_STEMMER = RSLPStemmer()
BOOLS = [True, False]
EXPERIMENTS = [(stem, case_folding, remove_stopwords) for stem in BOOLS for case_folding in BOOLS for remove_stopwords in BOOLS]
FEATURE_TYPES = ['BINARY', 'TF', 'LOG_TF', 'TF_IDF']
PUNCTUATION = ['.', ',', ':', ';', '(', ')', '[', ']', '*']
TEST_NUMBER = 0
METRICS_FILE = open(M_FILENAME, 'w')

def get_full_set():
	full_set = {}
	for genre in GENRES:
		full_set[genre] = list()

	genre_files = list()
	min_file_size = 5000

	data_path = os.getcwd() + "/lyrics_music/Data/lyrics/"
	for filename in os.listdir(data_path):
		filepath = data_path+filename
		cur_json = json.load(open(filepath, 'r'))
		genre_files.append(cur_json)
		min_file_size = min(min_file_size, len(cur_json))

	for genre_file in genre_files:
		filtered_genre_file = random.sample(genre_file, min_file_size)
		for item in filtered_genre_file:
			document = item['lyrics']
			full_set[item['genre']].append(document)
	return full_set

def create_sets(full_set, train_pct):
	train_set = {}
	test_set = {}
	for genre in full_set:
		train_set[genre] = list()
		test_set[genre] = list()

		train_sample = random.sample(full_set[genre], int(len(full_set[genre])*train_pct))
		test_sample = list(set(full_set[genre]) - set(train_sample))

		train_set[genre].extend(train_sample)
		test_set[genre].extend(test_sample)
	return (train_set, test_set)

def mean(arr):
	return np.mean(arr, dtype=np.float64)

def std_dev(arr):
	return np.std(arr, dtype=np.float64)

def stem_word(word):
	return PT_STEMMER.stem(word)

def remove_stopwords_from_doc(tokenized_doc):
	return filter(lambda word: word not in PT_STOPWORDS, tokenized_doc)

def lowercase_word(word):
	return word.lower()

def tokenize_doc(doc):
	return word_tokenize(doc)

def stem_tokenizer(doc):
	return filter(lambda t: t not in PUNCTUATION, [PT_STEMMER.stem(t) for t in word_tokenize(doc)])

def featurize_set(given_set, feature_type, stem, case_folding, remove_stopwords, vocabulary = None, max_df=800, min_df = 1, ngram_range=(1,2)):
	stop_list = None
	chosen_tokenizer = None
	if remove_stopwords:
		stop_list = PT_STOPWORDS
	if stem:
		chosen_tokenizer = stem_tokenizer

	documents = list()
	doc_genres = list()
	for genre in given_set:
		documents.extend(given_set[genre])
		doc_genres.extend([genre for i in range(len(given_set[genre]))])

	vectorizer = None
	if feature_type=='BINARY':
		vectorizer = TfidfVectorizer(lowercase=case_folding, tokenizer = chosen_tokenizer, stop_words=stop_list, binary = True, dtype = np.float64,
		 norm=None, use_idf=False, smooth_idf=False, vocabulary = vocabulary, max_df = max_df, min_df=min_df, ngram_range=ngram_range)
	elif feature_type =='TF':
		vectorizer = TfidfVectorizer(lowercase=case_folding, tokenizer = chosen_tokenizer, stop_words=stop_list, dtype = np.float64, norm=None, 
			use_idf=False, smooth_idf=False, vocabulary = vocabulary, max_df = max_df, min_df=min_df, ngram_range=ngram_range)
	elif feature_type =='LOG_TF':
		vectorizer = TfidfVectorizer(lowercase=case_folding, tokenizer = chosen_tokenizer, stop_words=stop_list, dtype = np.float64, norm=None,
		 use_idf=False, sublinear_tf=True, smooth_idf=False, vocabulary = vocabulary, max_df = max_df, min_df=min_df, ngram_range=ngram_range)
	else:
		vectorizer = TfidfVectorizer(lowercase=case_folding, tokenizer = chosen_tokenizer, stop_words=stop_list, dtype = np.float64, 
			vocabulary = vocabulary, max_df = max_df, min_df=min_df, ngram_range=ngram_range)

	featurized_docs_raw_format = vectorizer.fit_transform(documents)
	
	transformizer = DictVectorizer(dtype=np.float64).fit([vectorizer.vocabulary_])
	featurized_docs = transformizer.inverse_transform(featurized_docs_raw_format)
	featurized_set = [(featurized_docs[i], doc_genres[i]) for i in range(len(documents))]
	if vocabulary == None:
		return featurized_set, vectorizer.vocabulary_
	return featurized_set

def random_forest_model(num_of_trees=10, max_depth=None, num_of_cores = 1):
	return RandomForestClassifier(n_estimators=num_of_trees, max_depth=max_depth, n_jobs=num_of_cores)

def multinomial_naive_bayes_model(alpha=1.0, fit_prior=True):
	return MultinomialNB(alpha=alpha, fit_prior=fit_prior)

def logistic_regression_model(max_iter=100, multi_class='ovr', num_of_cores = 1, solver = 'liblinear'):
	return LogisticRegression(max_iter=max_iter, multi_class=multi_class, n_jobs=num_of_cores, solver=solver)

def linear_svc_model(max_iter=1000):
	return LinearSVC(max_iter=max_iter)

def svc_model(C=1.0,gamma='auto',kernel='rbf', degree=3, coef0=0.0, max_iter=-1, decision_function_shape='ovr'):
	return SVC(C=C, gamma=gamma,kernel=kernel, degree=degree, coef0=coef0, max_iter=max_iter, decision_function_shape=decision_function_shape)

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

	conf_matrix = ConfusionMatrix(reference_list, test_list)
	tp = conf_matrix[genre,genre]
	tn = sum([conf_matrix[i, i] for i in GENRES if i != genre])
	fn = sum([conf_matrix[genre, i] for i in GENRES if i != genre])
	fp = sum([conf_matrix[i, genre] for i in GENRES if i != genre])
	
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
	num_of_classes = float(len(genres_metrics))

	tp_sum = float(sum([genres_metrics[genre]['tp'] for genre in genres_metrics]))
	tn_sum = float(sum([genres_metrics[genre]['tn'] for genre in genres_metrics]))
	fp_sum = float(sum([genres_metrics[genre]['fp'] for genre in genres_metrics]))
	fn_sum = float(sum([genres_metrics[genre]['fn'] for genre in genres_metrics]))

	metrics['general'] = {}
	metrics['general']['accuracy'] = (tp_sum*2)/(2*tp_sum + fp_sum + fn_sum)

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
	metrics['micro']['accuracy'] = (tp_sum+tn_sum)/(tp_sum+tn_sum+fp_sum+fn_sum)
	metrics['micro']['precision'] = micro_precision
	metrics['micro']['recall'] = micro_recall
	metrics['micro']['f1'] = 2*micro_precision*micro_recall/(micro_precision+micro_recall)

	return metrics

def write_case_to_file(stem, case_folding, remove_stopwords, model_name, f_type):
	data = {}

	data['test_number'] = str(TEST_NUMBER)
	data['stemming'] = str(stem)
	data['case-folding'] = str(case_folding)
	data['remove_stopwords'] = str(remove_stopwords)
	data['clf'] = model_name
	data['feature'] = f_type
	return data

def write_metric_to_file(metric):
	data = {}

	data['tp'] = str(metric['tp'])
	data['tn'] = str(metric['tn'])
	data['fp'] = str(metric['fp'])
	data['fn'] = str(metric['fn'])
	data['accuracy'] = str(metric['accuracy'])
	data['precision'] = str(metric['precision'])
	data['recall'] = str(metric['recall'])
	data['f-measure'] = str(metric['f1'])

	return data

def write_genre_metrics_to_file(metrics):
	output = []

	for (genre, metric) in metrics.iteritems():
		data = {}
		data['genre'] = genre.encode('utf-8')
		data['metrics'] = write_metric_to_file(metric)
		output.append(data)
	return output

def write_classifier_metrics_to_file(metrics):
	output = []
	for (m_type, metric) in metrics.iteritems():
		data = {}
		if(m_type == 'general'):
			data[m_type] = metric
		else:
			data[m_type] = write_metric_to_file(metric)
		output.append(data)
	return output

def test(train_set, test_set, stem, case_folding, remove_stopwords):
	global TEST_NUMBER
	json_output = []

	for f_type in FEATURE_TYPES:
		#FEATURIZATION
		ready_train_set, vocabulary = featurize_set(train_set, f_type, stem, case_folding, remove_stopwords)
		ready_test_set = featurize_set(test_set, f_type, stem, case_folding, remove_stopwords, vocabulary)

		gold_test_labels = map(lambda (d,l): l, ready_test_set)
		test_documents = map(lambda (d,l): d, ready_test_set)

		for model_name in MODELS:
			start = time.time()
			TEST_NUMBER+=1

			#CLASSIFICATION
			clf = create_classifier(MODELS[model_name])
			trained_clf = train_classifier(clf, ready_train_set)
			predicted_labels = test_classifier(trained_clf, test_documents)

			#METRICS
			genres_metrics = {}
			for genre in GENRES:
				genres_metrics[genre] = save_genre_metrics(gold_test_labels, predicted_labels, genre)

			
			metrics = calculate_classifier_metrics(genres_metrics)
			json_data = {}
			json_data['test_details'] = write_case_to_file(stem, case_folding, remove_stopwords, model_name, f_type)
			json_data['genres_metrics'] = write_genre_metrics_to_file(genres_metrics)
			json_data['classifier_metrics'] = write_classifier_metrics_to_file(metrics)
			print json_data['test_details']
			print metrics['general']['accuracy']
			print metrics['macro']['f1']
			json_output.append(json_data)
			
	return json_output


MODELS = {'NAIVE BAYES DEFAULT': multinomial_naive_bayes_model(alpha=0.1), 
'LOGISTIC REGRESSION DEFAULT': logistic_regression_model(solver='sag',num_of_cores=2),
'SVM DEFAULT': svc_model(C=23.0, gamma = 0.001), 'LINEAR SVM DEFAULT': linear_svc_model(), 
'RANDOM FOREST DEFAULT': random_forest_model(num_of_trees = 100, num_of_cores=2)}

def experiment():
	#PRE-PROCESS
	full_set = get_full_set()
	train_set, test_set = create_sets(full_set, 0.7)
	merge_results = []
	#PROCESS
	for e in EXPERIMENTS:
		stem = e[0]
		case_folding = e[1]
		remove_stopwords = e[2]

		merge_results += test(train_set, test_set, stem, case_folding, remove_stopwords)
	json.dump(merge_results, METRICS_FILE)
	METRICS_FILE.close()

experiment()