import numpy as np
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk import word_tokenize

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

def create_sets(full_set, train_pct, test_pct, valid_pct = 0):
	train_set = [], test_set = [], valid_set = []
	#TODO
	return (train_set, test_set, valid_set)

def get_full_set():
	full_set = {}
	for genre in GENRES:
		full_set[genre] = get_docs_from_genre(genre)
	return full_set

def get_docs_from_genre(genre):
	#TODO
	return []