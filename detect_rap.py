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
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn import svm
from sklearn.manifold import TSNE

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
#features = ['total_num_words']
sfeatures = ['total_num_words','mean_word_length','rap']
a  = sns.pairplot(train_data.loc[:,sfeatures], hue='rap', diag_kind="kde", kind="reg")
sfeatures = ['median_word_rank','mean_word_instances','rap']
a  = sns.pairplot(train_data.loc[:,sfeatures], hue='rap', diag_kind="kde", kind="reg")
#a  = sns.pairplot(train_data.loc[:,features])
#%%
features = ['total_num_words','mean_word_length','median_word_rank','mean_word_instances']

g = train_data.groupby('rap')
print g.median()[features]
print g.sem()[features]
#%%
# Fit random forest to all the data and validate on the same data (not a good idea)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(train_data.loc[:,features],train_data.loc[:,'rap'])
prediction = clf.predict(train_data.loc[:,features])
prop_corr = np.mean(prediction == train_data.loc[:,'rap'])
print prop_corr
#%%
# Cross validate Random forest 
num_folds=10
num_instances = len(train_data)
scoring = 'accuracy'
seed = 7
clf = RandomForestClassifier(n_estimators=100)
kfold = cross_validation.KFold(n=num_instances, shuffle=True, n_folds=num_folds, random_state=seed)
results = cross_validation.cross_val_score(clf,train_data.loc[:,features], train_data.loc[:,'rap'], cv=kfold, scoring=scoring)
print np.mean(results)

#%%
# Cross validate Logistic regression
scaler = preprocessing.StandardScaler().fit(train_data.loc[:,features])
train_data_scaled = scaler.transform(train_data.loc[:,features])

clf = LogisticRegression()


results = cross_validation.cross_val_score(clf,train_data_scaled, train_data.loc[:,'rap'], cv=kfold, scoring=scoring)
print np.mean(results)
# Check a fit
splt = cross_validation.StratifiedShuffleSplit(train_data.loc[:,'rap'], test_size=0.1,n_iter=1, random_state=7)
splt = next(iter(splt))
fitinfo = clf.fit(train_data_scaled[splt[0],:], train_data.loc[splt[0],'rap'])
#%%
# Cross validate  SVM

scaler = preprocessing.StandardScaler().fit(train_data.loc[:,features])
train_data_scaled = scaler.transform(train_data.loc[:,features])

clf = svm.SVC()
results = cross_validation.cross_val_score(clf,train_data_scaled, train_data.loc[:,'rap'], cv=kfold, scoring=scoring)
print np.mean(results)

#%%
# Recursive number of features selection

#clf = svm.SVC(kernel='linear')
clf = RandomForestClassifier(n_estimators=100)
clf.fit(train_data_scaled, train_data.loc[:,'rap'])
rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(train_data.loc[:,'rap'], 2),
              scoring='accuracy')
rfecv.fit(train_data_scaled, train_data.loc[:,'rap'])

print("Optimal number of features : %d" % rfecv.n_features_)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

order = np.argsort(clf.feature_importances_)[::-1]
for i,o in enumerate(order):
	print "%d %s\tImportance = %.2f" % (i+1, features[o], clf.feature_importances_[o])
	
#%%
# TSNE embedding of the points onto a 2 plane
#
#
#tsne = TSNE()
#proj = tsne.fit_transform(train_data.loc[:,features])
#
#plt.figure() 
#plt.set_cmap("coolwarm")
#plt.scatter(proj[:, 0], proj[:, 1],s=10, c=train_data.loc[:,'rap'], alpha=0.5, edgecolors='face')
#plt.colorbar()
#

