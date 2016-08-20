
# coding: utf-8

# In[284]:

get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u"config InlineBackend.figure_formats = {'svg',}")

import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import confusion_matrix
from sklearn import metrics


# # Cross-Language Rap Detector
# A previous project of mine, RapItalia, was designed to approximately track the growth in popularity of rap in Italy, based on the publication date of rap songs that turned up in an Italian lyrics database. Did rap suddenly get popular there 10 years ago? I didn't have any genre information, only lyrical content, and so I made the assertion that rap songs could be distinguished by the number of words in the lyrics. I used a quite arbitrary cutoff for the number of words, specifically 500 words, and verified it only with a quick look at the songs that were identified as rap (I did made sure not to tune it based on the final output of my analysis). The results of this classification were then fed into my subsequent analysis of publication date by genre
# 
# Making an assumption like that without evidence is pretty sloppy, so I set out to do better. I did two things:
# 1. Linking a large body of lyrics with high quality genre information
# 2. Evaluated whether songs can be correctly classified as rap or non-rap based on non-language-specific properties of the lyrics (like the number of words)
# 
# Why non-language-specific? If I train on surface features of rap songs in every language, then I can identify rap songs in every language. This could also shed light generally on the surface-level lyrical differences between genres.
# 
# First, some helper functions:

# In[405]:

def feature_distribution(data, which_feature, xlim=None):
    med = np.median(train_data.loc[:,which_feature])
    q75, q25 = np.percentile(train_data.loc[:,which_feature], [75 ,25])
    iqr = q75 - q25
    minx = med-(iqr*2.5)
    if minx < 0:
        minx = 0
    maxx = med+(iqr*2.5)
    if xlim:
        minx=xlim[0]
        maxx=xlim[1]
        
    nbins = 20
    bins = np.linspace(minx, maxx, nbins+1) 

    plt.figure()
    sns.distplot(data.loc[data.is_rap==False,which_feature], bins=bins, label='Non-rap')
    sns.distplot(data.loc[data.is_rap==True,which_feature], bins=bins, label='Rap')
    plt.xlim(minx, maxx)
    plt.title(which_feature)
    plt.legend()
    
def plot_feature_importance(features, fitted_forest):
    plt.figure()
    vals = fitted_forest.feature_importances_
    sortorder = np.flipud(np.argsort(vals))
    features = np.array(features)
    with sns.axes_style("whitegrid"):
        sns.stripplot(y=features[sortorder], x=vals[sortorder], orient="h", color='red', size=10)
    xl = plt.xlim()
    plt.xlim(0,xl[1])
    plt.grid(axis='y',linestyle=':')
    plt.xlabel('Feature importance score')
    
def examine_prediction(y, prediction, data, features, show_misidentified=True):
    if type(features) == np.ndarray:
        features = features.tolist()
    
    cm = confusion_matrix(y, prediction)
    np.set_printoptions(precision=2)  
    nonrap_misidentified = float(cm[0,1])/(cm[0,0]+cm[0,1])
    print "Accuracy =\t%.1f%%" % (100*metrics.accuracy_score(y, prediction))
    print "Rap songs correctly identified =\t%.1f%%" % (100*metrics.recall_score(y, prediction))
    print "Songs incorrectly identified as rap =\t%.1f%%" % (100*(1-metrics.precision_score(y, prediction)))
    print "Non-rap songs identified as rap =\t%.1f%%" % (100*nonrap_misidentified)
    print "F1 score =\t%.3f" % metrics.f1_score(y, prediction)
    print('Confusion matrix')
    print(cm)
    if show_misidentified:
        print "Misidentified as rap: "
        display(data.loc[(prediction==1) & (y==0),['artist_name','title']+features])
        print "Misidentified as nonrap: "
        display(data.loc[(prediction==0) & (y==1),['artist_name','title']+features])

def compute_features(lyrics, tdm_indices):
    total_num_words = np.zeros(len(tdm_indices))
    for i in range(len(tdm_indices)):
        total_num_words[i] =  lyrics['tdm'][tdm_indices[i],:].sum()
	
    word_lens = np.array([len(i) for i in lyrics['unstemmed_terms']],dtype=float)
    mean_word_length = np.zeros(len(tdm_indices))
    for i in range(len(tdm_indices)):
        word_indices = lyrics['tdm'][tdm_indices[i],:].nonzero()[1]
        mean_word_length[i] = np.mean(word_lens[word_indices])

    median_word_rank = np.zeros(len(tdm_indices))
    for i in range(len(tdm_indices)):
        word_indices = lyrics['tdm'][tdm_indices[i],:].nonzero()[1]
        median_word_rank[i] = np.median(word_indices)
    
    mean_word_instances = np.zeros(len(tdm_indices))
    for i in range(len(tdm_indices)):
        nums = lyrics['tdm'][tdm_indices[i],:].toarray()
        nz = nums[nums.nonzero()]
        mean_word_instances[i] = np.mean(nz)
    mean_word_instances = np.divide(mean_word_instances, total_num_words)
    
    additional_features = pd.DataFrame(data={'total_num_words':total_num_words, 'mean_word_length':mean_word_length, 'median_word_rank':median_word_rank, 'mean_word_instances':mean_word_instances})
    return additional_features


# ## Creating the dataset
# My source for lyrics was the [musicXmatch Dataset](http://labrosa.ee.columbia.edu/millionsong/musixmatch), which contains entries for 237,662 songs from the Million Songs Dataset. The MSD is a selection of one million songs based on [loose criteria](http://labrosa.ee.columbia.edu/millionsong/pages/how-did-you-choose-million-tracks) that included as many songs as possible by popular artists, and "extreme" songs in terms of audio characteristics. However the complete lyrics are not included, for copyright reasons:
# 
# > The lyrics come in bag-of-words format: each track is described as the word-counts for a dictionary of the top 5,000 words across the set.
# 
# This eliminates at least two surface-level properties I was interested in, the line lengths and the occurrence of extremely rare (or made-up) words. But it retains many more. I stored lyrics information in a dict called lyrics, which has at the heart of it a sparse matrix of counts of words (columns) by tracks (rows), sorted in decreasing order of word frequency across the corpus.

# In[369]:

print lyrics['terms'][0:10]
print(lyrics['tdm'][:5,:].toarray())


# Although the Million Songs Dataset contains a large amount of metadata and data about the acoustic properties of songs (based on data compiled by [The Echo Nest](https://en.wikipedia.org/wiki/The_Echo_Nest), it does not have genre information. I got that from the [tagtraum genre annotations](www.tagtraum.com/msd_genre_datasets.html) to the Million Songs Dataset. It determines genre based on human-generated annotations from the All Music Guide, Last.fm, and the beaTunes Genre Dataset (BGD). There are up to two genres listed for every song, and I defined a track as being rap if it had "Rap" in either of the two genre slots. 
# 
# The tagtraum genre annotations covered 133,676 tracks, of which 55,726 intersected with the tracks in the musicXmatch lyrics training set, and 6,967 with the lyrics test set (the musicXmatch dataset has a standard train-test split). `generate_track_info.py` does this merge, and also adds track names and artist names by querying the MSD's sqlite3 database track_metadata.db, and saves the result as pickles.

# In[300]:

import pickle

with open('train_track_info.pickle','r') as f:
	track_info = pickle.load(f)
    
with open('train_lyrics_data.pickle','r') as f:
	lyrics = pickle.load(f)


# ## Feature engineering: Superficial text features
# Compute new features for each track based on the lyrics.

# In[331]:

new_features = compute_features(lyrics, track_info.tdm_row)
train_data = pd.concat([track_info, new_features],axis=1)
features = new_features.columns.values


# Examining the distribution of these variables between the two classes shows promising separation of tracks.
# 
# `total_num_words` is the number of words in the track, which will be an underestimate of the true number of words because of all words beyond the 5000 most frequent in the lyrics dataset being eliminated. Nevertheless, it should have a very strong linear correlation with the true number of words.

# In[316]:

# Create features
feature_distribution(train_data,'total_num_words',[0,1000])


# `mean_word_length` is the mean of the word lengths in a track, not weighting by frequency of the word. Again, not precisely the real values, since the lyrics have been stemmed (although I used the provided unstemming dictionary) but should correlate strongly.

# In[308]:

feature_distribution(train_data,'mean_word_length')


# `median_word_rank` is the median of the horizontal index of the words in the term-document matrix, which reflects the rarity of the words used.

# In[317]:

feature_distribution(train_data,'median_word_rank',[0,500])


# `mean_word_instances` is the mean number of times a word is repeated in a track, divided by the total number of words in the track. It should reflect how repetitive the song is lyrically (e.g. because of a high ratio of choruses to verses)

# In[310]:

feature_distribution(train_data,'mean_word_instances')


# ## How I Got to 95% Accuracy Without Really Trying: The Problem of Imbalanced Datasets
# All my initial attempts to correctly detect rap songs using the features I created seemed to be very successful: 95% accuracy. But then I realized that this was due to rap songs being much less common than non-rap.

# In[332]:

pd.value_counts(track_info.is_rap)


# In fact a dumb model that predicts that no songs will ever be rap achieves this accuracy, thanks to the imbalanced dataset.

# In[319]:

# Baseline model: none are rap
prediction = np.zeros(len(train_data))
print "Accuracy = %.1f%%" % (100* np.mean(prediction == train_data.is_rap))


# But this was very unsatisfactory for the puposes of my rap detector. I needed a) a better way to measure performance and b) a way to deal with training on this imbalanced data.
# 
# ## Imbalanced dataset therapy #1: Undersampling the classes to be equal
# First, for a metric that is relevant to the performance I care about, which includes correctly identifying rap as well as not incorrectly identifying songs as rap. (aka the recall and 1-the precision). I decided to focus on the F1 score, which combines the two. It correctly measures my rap-doesn't-exist baseline as terrible:

# In[320]:

examine_prediction(train_data.is_rap, prediction, train_data, features, show_misidentified=False)


# So I equalized the number of non-rap and rap tracks in my training set by selecting a random subset of the non-rap tracks.

# In[333]:

# Reduce the number of non-rap training samples so that it is balanced with the rap training samples
num_rap_tracks = np.sum(train_data.is_rap)
non_rap_tracks = train_data.loc[np.invert(train_data.is_rap),:]
rs = cross_validation.ShuffleSplit(len(non_rap_tracks), n_iter=1, test_size=num_rap_tracks,random_state=7)
sampled_nonrap_tracks =  next(iter(rs))[1]
non_rap_tracks = non_rap_tracks.iloc[sampled_nonrap_tracks,:]
train_data = pd.concat([non_rap_tracks,train_data.loc[train_data.is_rap,:]],ignore_index=True)

y = train_data.loc[:,'is_rap']

print "There are now %d non-rap tracks in the training set" % len(non_rap_tracks)


# In[ ]:

import scipy.sparse
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from sklearn.cross_validation import StratifiedKFold


# ### Random Forest fit
# With the non-rap and rap tracks equalized, and therefore the chance level of 50%, we can start training classifiers. Here's random forest, which did much better than chance, and also had a good F1 score:

# In[373]:

# Prepare for cross validation fittings

num_folds=5
num_instances = len(train_data)
scoring = 'accuracy'
seed = 7
kfold = cross_validation.KFold(n=num_instances, shuffle=True, n_folds=num_folds, random_state=seed)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100)
prediction = cross_validation.cross_val_predict(clf,train_data.loc[:,features], y, cv=kfold)

examine_prediction(y, prediction, train_data, features, show_misidentified=False)


# The next few algorithms require the features to be scaled (and I save this scaling so I can apply it to the test data):

# In[376]:

scaler = preprocessing.StandardScaler().fit(train_data.loc[:,features])
train_data_scaled = scaler.transform(train_data.loc[:,features])


# ### Logistic regression fit

# The logistic regression fit is almost as good, and much faster.

# In[378]:

# Cross validate Logistic regression
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
prediction = cross_validation.cross_val_predict(clf,train_data_scaled, y, cv=kfold)

examine_prediction(y, prediction, train_data, features, show_misidentified=False)


# ### Support vector machine fit
# The best performance (although probably not statistically reliably so) is from SVM

# In[379]:

# Cross validate SVM
from sklearn import svm

clf = svm.SVC()
prediction = cross_validation.cross_val_predict(clf,train_data_scaled, y, cv=kfold)

examine_prediction(y, prediction, train_data, features, show_misidentified=False)


# ### Feature importance and selection
# With the random forest classifier, we can use [recursive feature elimination with cross-validation](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV) to see how many features to use, and then to rank their importance.

# In[387]:

# Recursive feature selection and feature importance
from sklearn.feature_selection import RFECV

#clf = svm.SVC(kernel='linear')
clf = RandomForestClassifier(n_estimators=100)
rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(y, 3, random_state=7), scoring='f1')
rfecv.fit(train_data_scaled, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (f1)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)


# In[390]:

fitted_forest = clf.fit(train_data_scaled, y)
plot_feature_importance(features, fitted_forest)


# ### Embedding of the points in 2D (t-SNE)
# Just to get more insight into the separability of rap and non-rap using our features, I visualized the two classes embedded in 2D space using the [t-distributed stochastic neighbor embedding (t-SNE)](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding).

# In[ ]:

# t-SNE embedding of the points onto a 2 plane
from sklearn.manifold import TSNE

tsne = TSNE()
proj = tsne.fit_transform(train_data.loc[:,features])


# In[393]:

plt.figure() 
plt.set_cmap("coolwarm")
plt.scatter(proj[:, 0], proj[:, 1],s=2, c=y, alpha=1, edgecolors='face')


# ### Evaluating the undersampled models on the test set
# However, what really counts is performance on the test set. It's time to load that in, add the features, and then try it out with our trained models.

# In[421]:

with open('test_track_info.pickle','r') as f:
	test_track_info = pickle.load(f)
    
with open('test_lyrics_data.pickle','r') as f:
	test_lyrics = pickle.load(f)


# In[422]:

new_features = compute_features(test_lyrics, test_track_info.tdm_row)
test_data = pd.concat([test_track_info, new_features],axis=1)
test_data_scaled = scaler.transform(test_data.loc[:,features])  # Use scaler that was fit on train_data


# First up is SVM. Although the accuracy is still fairly high, and the F1 score is much higher than the no-rap baseline, the accuracy is actually lower, 93.1% vs 95.2%. Furthermore, the F1 score is lower than our cross-validation predicted, and full 46.8% of the songs identified as rap were not rap songs.

# In[406]:

# Train SVM on the whole training set
clf = svm.SVC()
fitinfo = clf.fit(train_data_scaled, y)
prediction = clf.predict(test_data_scaled)

examine_prediction(test_data.is_rap, prediction, test_data, features, show_misidentified=False)


# The random forest does even worse, with an accuracy below 90% and even more of the songs identified as rap being misclassified - in fact the majority.

# In[407]:

# Just for interest, a random forest
clf = RandomForestClassifier(n_estimators=100)
fitinfo = clf.fit(train_data_scaled, y)
prediction = clf.predict(test_data_scaled)

examine_prediction(test_data.is_rap, prediction, test_data, features, show_misidentified=False)


# Clearly, undersampling so that the dataset is 50/50 non-rap and rap songs biases the detector towards saying it is a rap song, which causes its performance to suffer on the test set, a mixed sample of tracks that contains less than 10% rap songs.
# ## Imbalanced dataset therapy #2: Training on unbalanced data with SVM class weights
# Next, I tried another option for imbalanced data sets, using the full training set but weighting my classifier by the appearances of each class in the training data. I focused on SVM, since it is faster than random forest and had some of the best performances.
# 
# Before trying the weighting, however, I decided to just try the whole training set without any class weighting.

# In[ ]:

un_train_data = track_info.copy()
new_features = compute_features(lyrics, un_train_data.tdm_row)
un_train_data = pd.concat([un_train_data, new_features],axis=1) 

# Cross-validation of SVM on unbalanced data
scaler_un = preprocessing.StandardScaler().fit(un_train_data.loc[:,features])
un_train_data_scaled = scaler_un.transform(un_train_data.loc[:,features])
un_y = un_train_data.is_rap

clf = svm.SVC()


# When I perform a cross-validation, the accuracy is higher, but the F1 score is lower than with the 50/50 trained models. We are only detecting about half of the rap songs.

# In[411]:

prediction = cross_validation.cross_val_predict(clf, un_train_data_scaled, un_y, cv=5)

examine_prediction(un_y, prediction, un_train_data, features, show_misidentified=False)


# Adding "balanced" class weighting (where classes are weighted based on the inverse of how often they appear) didn't make things better - although they greatly increased the number of rap songs that were identified as such, they made both accuracy and F1 worse.

# In[413]:

# Cross-validation of SVM on unbalanced data with class weightings
clf = svm.SVC(class_weight='balanced')
prediction = cross_validation.cross_val_predict(clf, un_train_data_scaled, un_y, cv=5)

examine_prediction(un_y, prediction, un_train_data, features, show_misidentified=False)


# I had a hunch that finding an intermediate level of class weighting, somewhere between none and balanced, would give me the best possible F1 on the test set. I started by recreating manually the balanced class weightings.

# In[414]:

# Recreate the class weightings that 'balanced' produces
cw = len(un_y)/(2. * np.bincount(un_y))
print cw


# Then I used grid search cross validation to take 10 steps between completely unweighted and weighted, to find the class weights that optimize F1.

# In[417]:

# Search to optimize the class weightings, with 10 steps between no weighting and balanced weighting
from sklearn import grid_search
import time

ratios = np.linspace(float(np.sum(un_y))/len(un_y),0.5,10)
cws = [{0:len(un_y)/(2 * len(un_y)*(1-r)),1:len(un_y)/(2 * len(un_y)*r)} for r in ratios]

start = time.time()
param_grid = {'class_weight':cws}
clf = svm.SVC()
gs = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, verbose=3, scoring='f1',n_jobs=-1)
gs.fit(un_train_data_scaled, un_y)
for params, mean_score, scores in gs.grid_scores_:
       print("%0.3f (+/-%0.03f) for %r"
            % (mean_score, scores.std() * 2, params))
print time.time()-start


# ```0.388 (+/-0.007) for {'class_weight': {0: 0.51978927203065139, 1: 13.133107454017425}}
# 0.502 (+/-0.068) for {'class_weight': {0: 0.54908675542404095, 1: 5.5930235221364608}}
# 0.532 (+/-0.121) for {'class_weight': {0: 0.58188416502647344, 1: 3.5530933535094866}}
# 0.533 (+/-0.127) for {'class_weight': {0: 0.61884850189686236, 1: 2.6035183112106193}}
# 0.532 (+/-0.144) for {'class_weight': {0: 0.66082774969420788, 1: 2.0544581110868068}}
# 0.528 (+/-0.158) for {'class_weight': {0: 0.70891672593732302, 1: 1.6966490422361025}}
# 0.511 (+/-0.146) for {'class_weight': {0: 0.76455394210358241, 1: 1.4449868635944048}}
# 0.483 (+/-0.149) for {'class_weight': {0: 0.829667958862641, 1: 1.2583387868887939}}
# 0.452 (+/-0.128) for {'class_weight': {0: 0.90690549052231262, 1: 1.1143932825263543}}
# 0.390 (+/-0.077) for {'class_weight': {0: 1.0, 1: 1.0}}```

# In[423]:

clf = svm.SVC(class_weight=cws[5])
test_data_scaled_un = scaler_un.transform(test_data.loc[:,features]) # Different scaler
fitinfo = clf.fit(un_train_data_scaled, un_y)
prediction = clf.predict(test_data_scaled_un)

examine_prediction(test_data.is_rap, prediction, test_data, features, show_misidentified=True)


# In[ ]:



