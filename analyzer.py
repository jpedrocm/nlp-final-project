import json

def get_data():
	with open('metrics_file.json') as data_file:    
		data = json.load(data_file)
		return data

def filter_strategies(given_list, stem=None, rmv_stopwords=None, folding=None, feature=None, clf=None):
	filtered_list = given_list
	if stem != None:
		filtered_list = filter(lambda d: d['test_details']['stemming']==str(stem), given_list)
	if rmv_stopwords != None:
		filtered_list = filter(lambda d: d['test_details']['remove_stopwords']==str(rmv_stopwords), given_list)
	if folding != None:
		filtered_list = filter(lambda d: d['test_details']["case-folding"]==str(folding), given_list)
	if feature != None:
		filtered_list = filter(lambda d: d['test_details']['feature']==feature, given_list)
	if clf != None:
		filtered_list = filter(lambda d: d['test_details']['clf']==clf, given_list)
	return filtered_list

def sort_strategies(given_list, decreasing):
	return sorted(given_list, key=lambda k: k['classifier_metrics']['general']['accuracy'], reverse=decreasing)

def analyze():
	data = get_data()
	data_filt = filter_strategies(data)
	data_sort = sort_strategies(data_filt, decreasing=True)
	for item in data_sort:
		print item[test_details]

analyze()