Cross-Language Rap Detector
---------------------------

For my RapItalia project, I made the claim that I could identify both Italian and English rap songs in my lyrics dataset based only on the number of words. Therefore I decided to use machine learning and a labelled dataset to find out if this is the case.

This is a project to learn features that will accurately identify rap songs based on their lyrics, without using semantic or syntactic information, only surface-level properties of the word list. It builds a training set from the [musicXmatch Dataset](http://labrosa.ee.columbia.edu/millionsong/musixmatch), with genre information provided by Spotify. It consists of three parts:

* **`dump_track_mappings.py`**, which dumps data out of a [newly-created database](http://labs.acousticbrainz.org/million-song-dataset-mapping) by the [AcousticBrainz Labs](http://labs.acousticbrainz.org/) that can map between musicXmatch IDs and both Spotify artist IDs and Spotify track Ids, the output of which is included as `musix_to_spotify_artists.json` and `musix_to_spotify_tracks.json`.
* **`generate_track_info.py`**, which uses those mappings to query the [Spotify web API](https://developer.spotify.com/web-api/) for track and artist info for each track in the musicXmatch database, and in particular to score it as rap or not. Then, it loads the lyrics data for only those songs that have . The output of those steps is stored as `train_track_info.pickle` and `train_lyrics_data.pickle` and the equivalent for the test set.
* **`detect_rap.ipynb`**, the Jupyter notebook where I generate my lyrics features, and use them to train and evaluate a rap song classifier. 