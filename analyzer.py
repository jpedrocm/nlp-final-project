import json

def get_data():
	with open('metrics_file.json') as data_file:    
		data = json.load(data_file)
		return data

def filter_strategies(given_list, stem=None, rmv_stopwords=None, folding=None, feature=None, clf=None):
	filtered_list = given_list
	if stem != None:
		filtered_list = filter(lambda d: d['test_details']['stemming']==str(stem), filtered_list)
	if rmv_stopwords != None:
		filtered_list = filter(lambda d: d['test_details']['remove_stopwords']==str(rmv_stopwords), filtered_list)
	if folding != None:
		filtered_list = filter(lambda d: d['test_details']["case-folding"]==str(folding), filtered_list)
	if feature != None:
		filtered_list = filter(lambda d: d['test_details']['feature']==feature, filtered_list)
	if clf != None:
		filtered_list = filter(lambda d: d['test_details']['clf']==clf, filtered_list)
	return filtered_list

def sort_strategies(given_list, decreasing):
	return sorted(given_list, key=lambda k: k['classifier_metrics'][2]['general']['accuracy'], reverse=decreasing)

def analyze():
	data = get_data()
	data_filt = filter_strategies(data, stem = None, rmv_stopwords = None, folding = None, feature = None, clf = "LOGISTIC REGRESSION DEFAULT")
	data_sort = sort_strategies(data_filt, decreasing=True)
	fi = open('analysis.txt', 'w')
	for item in data_sort:
		fi.write(str(item['test_details']))
		fi.write(str(item['classifier_metrics'][2]['general']['accuracy']))
		fi.write('\n')
	fi.close()

analyze()