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
import time
import re
#%%

#%%
def load_tids(lyrics_file):
    tids = []
    with open(lyrics_file, 'r') as f:
        for line in f.xreadlines():
            if line == '' or line.strip() == '':
                continue
            if line[0] in ('#', '%'):
                continue
            lineparts = line.strip().split(',')
            tids.append(lineparts[0])

    return np.array(tids)
#%%
# Load lyrics from musicXmatch training set
def load_lyrics(lyrics_file, tids):
    lyrics = {}

    # Enable fast lookup (hash based) of tids
    tids = set(tids)
    
    # Read in the list of terms at the beginning of the file
    with open(lyrics_file, 'r') as f:    
        for line in f:
            if line == '' or line.strip() == '':
                continue
            if line[0] in ('%'):
                lyrics['terms'] = line.strip()[1:].split(',')
                break
                                
    # Figure out the unstemmed versions of all the terms (note that this will 
     # not necessarily recover the originals since multiple words can be converted
     # to the same stem)
    stemmed_to_unstemmed = pd.read_csv('musicXmatch/mxm_reverse_mapping.txt',sep='<SEP>',header=None,index_col=0,names=['unstemmed'])
    lyrics['unstemmed_terms'] = stemmed_to_unstemmed.loc[lyrics['terms']].unstemmed.values
                    
#    tdm = pd.read_csv(lyrics_file, skiprows=18, nrows=1000)


    tdm = np.zeros([len(tids),len(lyrics['terms'])], dtype=np.uint32)
    lyrics['track_id'] = []
    start = time.time()
    with open(lyrics_file, 'r') as f:
        cnt_lines = 0
        for line in f:
            if line == '' or line.strip() == '':
                continue
            if line[0] in ('#', '%'):
                continue
            
            lineparts = re.split(r"[:,]",line)
            tid = lineparts[0]
            if not (tid in tids):
                  continue

            indices = np.uint16(lineparts[2::2])
            counts = np.uint16(lineparts[3::2])
            tdm[cnt_lines,indices-1] = counts

#            
            lyrics['track_id'].append(lineparts[0])
#            
#            for wordcnt in lineparts[2:]:
#                wordid, cnt = wordcnt.split(':')
#                tdm[cnt_lines,int(wordid)-1] = int(cnt)
                
        #        print "%s\t%s" % (wordid,cnt)            indices = 
            cnt_lines += 1
            if cnt_lines % 1000 == 0:
                print (time.time() - start) / 60.
                print 'Done with %d tracks.' % cnt_lines
    lyrics['tdm'] = scipy.sparse.csr_matrix(tdm)
    print (time.time() - start) / 60.
    return lyrics
#%%
def lookup_track_info(tids):
    import sqlite3
    con = sqlite3.connect('track_metadata.db')
    all_songs = pd.read_sql('select * from songs',con)
    con.close()
    track_info = all_songs.loc[all_songs.track_id.isin(tids),['track_id','title','artist_name','year']]

    tagtraum = pd.read_csv('msd_tagtraum_cd1.cls',sep='\t',comment='#',names=['id','genre1','genre2'])
    tagtraum = tagtraum.fillna('')
    track_info = track_info.merge(tagtraum, how='inner',left_on='track_id',right_on='id')
    track_info.loc[:,'is_rap'] = (track_info.loc[:,'genre1'] == 'Rap') | (track_info.loc[:,'genre2'] == 'Rap')
    
    return track_info
#%%
# Start with a basic list of all the track ids in the musicXmatch lyrics database
tids = load_tids('musicXmatch/mxm_dataset_train.txt')

track_info = lookup_track_info(tids)

lyrics = load_lyrics('musicXmatch/mxm_dataset_train.txt', track_info.track_id.values)

# Add the tdm_row, mapping the track to the row of the term-document matrix in lyrics
track_info = track_info.merge(pd.DataFrame({'tdm_row':range(len(lyrics['track_id'])),'track_id':lyrics['track_id']}),on='track_id')

#%%
# Save the generated data structures in pickle format
with open('train_track_info.pickle','w') as f:
    pickle.dump(track_info,f)
    #%%
with open('train_lyrics_data.pickle','w') as f:
    pickle.dump(lyrics,f)    
    
#%%
# Start with a basic list of all the track ids in the musicXmatch lyrics database
tids = load_tids('musicXmatch/mxm_dataset_test.txt')

track_info = lookup_track_info(tids)

lyrics = load_lyrics('musicXmatch/mxm_dataset_test.txt', track_info.track_id.values)

# Add the tdm_row
track_info = track_info.merge(pd.DataFrame({'tdm_row':range(len(lyrics['track_id'])),'track_id':lyrics['track_id']}),on='track_id')

#%%
# Save the generated data structures in pickle format
with open('test_track_info.pickle','w') as f:
    pickle.dump(track_info,f)
#%%
with open('test_lyrics_data.pickle','w') as f:
    pickle.dump(lyrics,f)
    #%%
with open('justtdm.pickle','w') as f:
    pickle.dump(lyrics['tdm'],f)
#%%
with open('notsparse.pickle','w') as f:
    pickle.dump(lyrics['tdm'].toarray(),f)
#%%

lil = scipy.sparse.lil_matrix(lyrics['tdm'])
with open('lil.pickle','w') as f:
    pickle.dump(lil,f)
