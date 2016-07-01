# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 17:16:58 2016

@author: dsaunder
"""

import os
import json

rootdir =  '/Users/dsaunder/IN/millionsongdataset_echonest/'
twoletters = os.listdir(rootdir)

spotify_track_mapping= {}
spotify_artist_mapping = {}
for tl in twoletters:
	print tl
	if len(tl) > 2:
		continue
	
	songfiles = os.listdir(rootdir + tl)
	for sf in songfiles:
		if '.json' in sf:
			f = open(rootdir + tl + os.sep + sf)
			response = json.load(f)['response']
			if len(response['songs']) > 0:
				songdata = response['songs'][0]
				for track in songdata['tracks']:
					if track['catalog'] == 'spotify':
						spotify_id = track['foreign_id'].split(':')[2]
						spotify_track_mapping[track['id']] = spotify_id
				if 'artist_foreign_ids' in songdata:
					for artist_foreign_ids in songdata['artist_foreign_ids']:
						if artist_foreign_ids['catalog'] == 'spotify':
							spotify_id = artist_foreign_ids['foreign_id'].split(':')[2]
							spotify_artist_mapping[track['id']] = spotify_id
					
			f.close()
#				spotify_track_mapping[track['id']] = 
	