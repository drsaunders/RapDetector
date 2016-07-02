# -*- coding: utf-8 -*-
"""
Created on Fri May 27 22:15:45 2016

@author: dsaunder
"""

import numpy as np
import pandas as pd
import pickle
import scipy.sparse
from sklearn import cross_validation
#%%
def dumpWordsForTrack(lyrics,whichRow):
	print lyrics['mxm_tids'][whichRow]
	track_lyrics = lyrics['tdm'][whichRow,:].toarray()[0]
	order = np.argsort(track_lyrics)[::-1]
	order = order[track_lyrics[order] > 0]
	words = np.array(lyrics['unstemmed_terms'])[order]
	print ' '.join(words)
	print track_lyrics[order]
	print np.sum(track_lyrics)
#%%
with open('test_track_info.pickle','r') as f:
	track_info = pickle.load(f)
#%%
with open('test_lyrics_data.pickle','r') as f:
	lyrics = pickle.load(f)

#%%
# Create the training data, including features
y = track_info.loc[:,'rap']
num_rap_tracks = np.sum(y)
all_rap_artists = np.unique(track_info.loc[track_info.rap==1,'spotify_artist_id'])
all_nonrap_artists = np.unique(track_info.loc[track_info.rap==0,'spotify_artist_id'])
#%%
g=  track_info.groupby(by=['rap','spotify_artist_id']) 
num_songs_per_artist = g.count()['mxm_tid']
rap_songs_per_artist = np.mean(num_songs_per_artist[1])
non_rap_songs_per_artist = np.mean(num_songs_per_artist[0])
#%%
#  Select a non-rap sample that  matches the number of rap tracks, but also has

rs = cross_validation.ShuffleSplit(len(all_nonrap_artists), n_iter=1, test_size=len(all_rap_artists),random_state=7)
sampled_nonrap_artists =  next(iter(rs))[1]
non_rap_tracks = track_info.merge(right=pd.DataFrame(data={'spotify_artist_id':all_nonrap_artists[sampled_nonrap_artists]}),on='spotify_artist_id',how='inner')
rs = cross_validation.ShuffleSplit(len(non_rap_tracks), n_iter=1, test_size=num_rap_tracks,random_state=7)
sampled_nonrap_tracks =  next(iter(rs))[1]
non_rap_tracks = non_rap_tracks.loc[sampled_nonrap_tracks,:]
train_data = pd.concat([non_rap_tracks,track_info.loc[track_info.rap==1,:]],ignore_index=True)

#%%

def spotCheck(lyrics,train_data):
	#%%
	n = 5
	c = []
	s1 = train_data[train_data['rap'] == 1].index.values
	np.random.shuffle(s1)
	s2 = train_data[train_data['rap'] == 0].index.values
	np.random.shuffle(s2)
	
	s = np.concatenate([s1[:n],s2[:n]])
	
	b = []
	for i in s:
		print "%s\t%s" % (train_data.loc[i,'track_name'],train_data.loc[i,'names'])
		dumpWordsForTrack(lyrics, train_data.tdm_row[i])
		b.append(lyrics['tdm'][train_data.tdm_row[i],:].sum(b))
	#%%
	
# Create features
total_num_words = np.zeros(len(train_data))
for i in range(len(train_data)):
	total_num_words[i] =  lyrics['tdm'][train_data.tdm_row[i],:].sum()
	
train_data.loc[:,'total_num_words'] = total_num_words

word_lens = np.array([len(i) for i in lyrics['unstemmed_terms']],dtype=float)
mean_word_length = np.zeros(len(train_data))
for i in range(len(train_data)):
	word_indices = lyrics['tdm'][train_data.tdm_row[i],:].nonzero()[1]
	mean_word_length[i] = np.mean(word_lens[word_indices])
train_data.loc[:,'mean_word_length'] = mean_word_length
#%%
median_word_rank = np.zeros(len(train_data))
for i in range(len(train_data)):
	word_indices = lyrics['tdm'][train_data.tdm_row[i],:].nonzero()[1]
	median_word_rank[i] = np.median(word_indices)
train_data.loc[:,'median_word_rank'] = median_word_rank
#%%
mean_word_instances = np.zeros(len(train_data))
for i in range(len(train_data)):
	nums = lyrics['tdm'][train_data.tdm_row[i],:].toarray()
	nz = nums[nums.nonzero()]
	mean_word_instances[i] = np.mean(nz)
train_data.loc[:,'mean_word_instances'] = mean_word_instances

#train_data.loc[:,'weighted_median_word_rank'] = 

#%%
import seaborn as sns
#features = ['total_num_words']
features = ['total_num_words','mean_word_length','rap']
a  = sns.pairplot(train_data.loc[:,features], hue='rap', diag_kind="kde", kind="reg")
features = ['median_word_rank','mean_word_instances','rap']
a  = sns.pairplot(train_data.loc[:,features], hue='rap', diag_kind="kde", kind="reg")
#a  = sns.pairplot(train_data.loc[:,features])
#%%
g = train_data.groupby('rap')
print g.median()[features[:-1]]
print g.sem()[features[:-1]]
#%%
from sklearn.ensemble import RandomForestClassifier
features = ['total_num_words','mean_word_length','median_word_rank','mean_word_instances']
#features = ['mean_word_length','median_word_rank','mean_word_instances']
#features = ['total_num_words']


#from sklearn.preprocessing import Imputer\
clf = RandomForestClassifier(n_estimators=100)
clf.fit(train_data.loc[:,features],train_data.loc[:,'rap'])
prediction = clf.predict(train_data.loc[:,features])
prop_corr = np.mean(prediction == train_data.loc[:,'rap'])
print prop_corr
#%%
from sklearn import cross_validation
num_folds=10
num_instances = len(train_data)
scoring = 'accuracy'
seed = 7
kfold = cross_validation.KFold(n=num_instances, shuffle=True, n_folds=num_folds, random_state=seed)
results = cross_validation.cross_val_score(clf,train_data.loc[:,features], train_data.loc[:,'rap'], cv=kfold, scoring=scoring)
print np.mean(results)

#%%
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
results = cross_validation.cross_val_score(clf,train_data.loc[:,features], train_data.loc[:,'rap'], cv=kfold, scoring=scoring)
print np.mean(results)

#%%
from sklearn import svm
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(train_data.loc[:,features])
train_data_scaled = scaler.transform(train_data.loc[:,features])

clf = svm.SVC()
results = cross_validation.cross_val_score(clf,train_data_scaled, train_data.loc[:,'rap'], cv=kfold, scoring=scoring)
print np.mean(results)
#%%

# TSNE embedding
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE()
proj = tsne.fit_transform(train_data.loc[:,features])

plt.figure() 
plt.set_cmap("coolwarm")
plt.scatter(proj[:, 0], proj[:, 1],s=10, c=train_data.loc[:,'rap'], alpha=0.5, edgecolors='face')
plt.colorbar()


#%%
#
#
#
##%%
#f = open('musicXmatch/mxm_779k_matches.txt','r')
#track_artist_info = pd.DataFrame(columns=['tid','artistname','rap'])
#for line in f:
#	if line == '' or line.strip() == '':
#		continue
#	if line[0] in ('#', '%'):
#		continue
#	
#	lineparts = line.strip().split('<SEP>')
#	msdid = lineparts[0]
#	artistname = lineparts[1]
#	if msdid in tids:
#		track_artist_info[]
#	break
#f.close()
##%%
#
#import sqlite3
#import pandas as pd
#import numpy as np
#
#con = sqlite3.connect('artist_term.db')
#artist_tags = pd.read_sql('SELECT * FROM artist_mbtag',con)
#artist_term = pd.read_sql('SELECT * FROM artist_term',con)
#
#termrap = np.array((artist_term.term == "rap") | (artist_term.term == "rapping") | (artist_term.term == "hiphop") |(artist_term.term == "hip hop"), dtype="int")
#artist_term.loc[:,'rap'] = termrap
#g = artist_term.groupby('artist_id')
#artist_raps = pd.DataFrame(g['rap'].max())
##%%
##for aid in artist_raps.loc[artist_raps.rap==1].index:
##    print(artist_term.loc[artist_term.artist_id==aid,:])
#    
#
#mbtagrap = np.array((artist_tags.mbtag == "rap") | (artist_tags.mbtag == "rapping") | (artist_tags.mbtag == "hiphop") |(artist_tags.mbtag == "hip hop"), dtype="int")
#artist_tags.loc[:,'rap'] = mbtagrap
#g = artist_tags.groupby('artist_id')
#artist_raps2 = pd.DataFrame(g['rap'].max())
#
#a = artist_raps.merge(artist_raps2,how='outer',left_index=True,right_index=True)
#
#artist_raps = pd.DataFrame({'artist_id':a.index, 'raps':(a.rap_x==1) | (a.rap_y==1)})
#
#con.close()
##%%
#
#con = sqlite3.connect('track_metadata.db')
#sql = 'SELECT track_id, artist_id FROM songs'
#artist_tracks = pd.read_sql(sql,con)
#
#rap_tracks = artist_tracks.merge(artist_raps,how='left',on='artist_id')
#con.close()
