# coding: utf-8

import os, json

GENRES = ["Sertanejo", "Funk Carioca", u"Axé",  "MPB", "Samba"]

def get_full_set():
	full_set = {}
	for genre in GENRES:
		full_set[genre] = list()

	genre_files = list()

	data_path = os.getcwd() + "/lyrics_music/Data/lyrics/"
	for filename in os.listdir(data_path):
		filepath = data_path+filename
		cur_json = json.load(open(filepath, 'r'))
		genre_files.append(cur_json)

	for genre_file in genre_files:
		for item in genre_file:
			artist = item['link'].split('/')[3]
			if artist not in full_set[item['genre']]:
				full_set[item['genre']].append(artist)
	return full_set

full_set = get_full_set()
for genre in full_set:
	print genre
	print full_set[genre]
	print len(full_set[genre])