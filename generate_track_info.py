# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 20:06:14 2016

@author: dsaunder
"""

import numpy as np
import pandas as pd
import spotipy
import json
import pickle
import scipy.sparse

#%%

#%%
def load_mxm_tids(lyrics_file):
	mxm_tids = []
	f = open(lyrics_file, 'r')
	for line in f.xreadlines():
	    if line == '' or line.strip() == '':
	        continue
	    if line[0] in ('#', '%'):
	        continue
	    lineparts = line.strip().split(',')
	    mxm_tids.append(lineparts[1])

	return np.array(mxm_tids)
#%%
# Load lyrics from musicXmatch training set
def load_lyrics(lyrics_file, mxm_tids):
	lyrics = {}
	mxm_tids = set(mxm_tids)

	# Read in the list of terms at the beginning of the file
	f = open(lyrics_file, 'r')	
	for line in f.xreadlines():
	    if line == '' or line.strip() == '':
	        continue
	    if line[0] in ('%'):
	        lyrics['terms'] = line.strip()[1:].split(',')
	        f.close()
	        break
								
	# Figure out the unstemmed versions of all the terms
	stemmed_to_unstemmed = pd.read_csv('musicXmatch/mxm_reverse_mapping.txt',sep='<SEP>',header=None,index_col=0,names=['unstemmed'])
	lyrics['unstemmed_terms'] = stemmed_to_unstemmed.loc[lyrics['terms']].unstemmed.values
					
	tdm = np.zeros([len(mxm_tids),len(lyrics['terms'])], dtype=np.uint8)
	lyrics['tids'] = []
	lyrics['mxm_tids'] = []
	
	f = open(lyrics_file, 'r')
	cnt_lines = 0
	for line in f.xreadlines():
	    if line == '' or line.strip() == '':
	        continue
	    if line[0] in ('#', '%'):
	        continue
	    lineparts = line.strip().split(',')
	    mxm_tid = lineparts[1]
	    if not (mxm_tid in mxm_tids):
              continue
		
	    lyrics['mxm_tids'].append(mxm_tid)
	    lyrics['tids'].append(lineparts[0])
	    for wordcnt in lineparts[2:]:
	        wordid, cnt = wordcnt.split(':')
	        tdm[cnt_lines,int(wordid)-1] = int(cnt)
	        
	#        print "%s\t%s" % (wordid,cnt)
	    cnt_lines += 1
	    if cnt_lines % 15000 == 0:
	        print 'Done with %d tracks.' % cnt_lines
	f.close()
	lyrics['tdm'] = scipy.sparse.csr_matrix(tdm)
	
	return lyrics
#%%
# For each mxm track id passed in, look up all the info about that track on spotify
def lookup_all_track_info(mxm_tids):
	f = open('musix_to_spotify_artists.json')
	musix_to_spotify_artists = json.load(f)
	f.close()
	f = open('musix_to_spotify_tracks.json')
	musix_to_spotify_tracks = json.load(f)
	f.close()
	
	# Map musicxmatch tracks to spotify artists, noting where one isn't available
	artist_available = np.zeros(len(mxm_tids))
	spotify_artists = []
	for i,tid in enumerate(mxm_tids): 
		if tid in musix_to_spotify_artists:
			spotify_artists.append(musix_to_spotify_artists[tid])
			artist_available[i] = 1
		else:
			spotify_artists.append('')

	# Map musicxmatch tracks to spotify tracks, noting where one isn't available
	track_available = np.zeros(len(mxm_tids))
	spotify_tracks = []
	for i,tid in enumerate(mxm_tids): 
		if tid in musix_to_spotify_tracks:
			spotify_tracks.append(musix_to_spotify_tracks[tid])
			track_available[i] = 1
		else:
			spotify_tracks.append('')

	inlookup =  (artist_available==1) & (track_available ==1)
		
	print np.sum(inlookup)
	track_info = pd.DataFrame(data={'mxm_tid':mxm_tids[inlookup], \
	'spotify_artist_id':np.array(spotify_artists)[inlookup], \
	'spotify_track_id':np.array(spotify_tracks)[inlookup]})
	
	
	unique_artists = np.unique(track_info.spotify_artist_id)

	ids = []
	names = []
	rap = []
	i = 0
	spotify = spotipy.Spotify()

	# Look up all the artists in spotify, 
	for starti in range(0,len(unique_artists),50):
		lasti = starti + 50
		if lasti > len(unique_artists):
			lasti = len(unique_artists)
		print "Querying spotify for artists %d to %d" % (starti, lasti-1)
		response = spotify.artists(unique_artists[starti:lasti])['artists']
		for artist_data in response:
			genres = artist_data['genres']
			if len(genres) > 0:
				names.append(artist_data['name'])
				israp = False
				for g in genres:
					if ('rap' in g) or ('hip hop' in g):
						israp = True
				rap.append(israp)
	#			if artist_data['id'] != unique_artists[i]:
	#				print  '\n\n' + unique_artists[i] + '\n'
	#				print artist_data
				ids.append(unique_artists[i])  #sometimes artists have more than one id that maps to them
			i = i + 1
			
	artist_info = pd.DataFrame({'spotify_artist_id':ids,'names':names,'rap':rap})

	track_info = track_info.merge(artist_info,on='spotify_artist_id',how='inner')
	track_info.rap = track_info.rap.astype(int)
	
	track_names = []
	for starti in range(0,len(track_info), 50):
		lasti = starti + 50
		if lasti > len(track_info):
			lasti = len(track_info)
		print "Querying spotify for tracks %d to %d" % (starti, lasti-1)
		response = spotify.tracks(track_info.loc[starti:(lasti-1),'spotify_track_id'])['tracks']
		for track_data in response:
			if track_data:
				if 'name' in track_data:
					track_names.append(track_data['name'])
				else:
					track_names.append('')
			else:
				track_names.append('')
	track_info.loc[:,'track_name'] = np.array(track_names)		

	return track_info
#%%
# Start with a basic list of all the track ids in the musicXmatch lyrics database
mxm_tids = load_mxm_tids('musicXmatch/mxm_dataset_train.txt')

track_info = lookup_all_track_info(mxm_tids)

# The row in the lyrics term-document matrix will be contiguous, because we
# are about to filter out only the mxm_tids present in the track info.


lyrics = load_lyrics('musicXmatch/mxm_dataset_train.txt', track_info.mxm_tid.values)
track_info = track_info.merge(pd.DataFrame({'tdm_row':range(len(lyrics['mxm_tids'])),'mxm_tid':lyrics['mxm_tids']}),on='mxm_tid')

#%%
# Save the generated data structures in pickle format
with open('train_track_info.pickle','w') as f:
	pickle.dump(track_info,f)
#%%
with open('train_lyrics_data.pickle','w') as f:
	pickle.dump(lyrics,f)