import json
from pprint import pprint

import spotipy
import time
import pandas as pd
from spotipy import SpotifyClientCredentials, util


client_id = '9090ef272da14bdab9e31501c0570ca2'
client_secret = '3564640bad7843ec9ca947ad570ce855'
redirect_uri = 'http://localhost:8888/callback'
username = 'pavan.srivathsav@gmail.com'
scope = 'playlist-modify-public'

def connectSpotifyClient():
    # Credentials to access the Spotify Music Data
    manager = SpotifyClientCredentials(client_id, client_secret)
    sp = spotipy.Spotify(client_credentials_manager=manager)

    # Credentials to access to  the Spotify User's Playlist, Favorite Songs, etc.
    token = util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri)
    spt = spotipy.Spotify(auth=token)
    return sp, spt



def get_albums_id(sp, ids):
    album_ids = []
    results = sp.artist_albums(ids)
    for album in results['items']:
        album_ids.append(album['id'])
    return album_ids


def get_album_songs_id(sp, ids):
    song_ids = []
    results = sp.album_tracks(ids, offset=0)
    for songs in results['items']:
        song_ids.append(songs['id'])
    return song_ids


def get_songs_features(sp, ids):
    meta = sp.track(ids)
    features = sp.audio_features(ids)

    # meta
    name = meta['name']
    album = meta['album']['name']
    artist = meta['album']['artists'][0]['name']
    release_date = meta['album']['release_date']
    length = meta['duration_ms']
    popularity = meta['popularity']
    ids = meta['id']

    # features
    acousticness = features[0]['acousticness']
    danceability = features[0]['danceability']
    energy = features[0]['energy']
    instrumentalness = features[0]['instrumentalness']
    liveness = features[0]['liveness']
    valence = features[0]['valence']
    loudness = features[0]['loudness']
    speechiness = features[0]['speechiness']
    tempo = features[0]['tempo']
    key = features[0]['key']
    time_signature = features[0]['time_signature']

    track = [name, album, artist, ids, release_date, popularity, length, danceability, acousticness,
             energy, instrumentalness, liveness, valence, loudness, speechiness, tempo, key, time_signature]
    columns = ['name', 'album', 'artist', 'id', 'release_date', 'popularity', 'length', 'danceability', 'acousticness',
               'energy', 'instrumentalness',
               'liveness', 'valence', 'loudness', 'speechiness', 'tempo', 'key', 'time_signature']
    return track, columns


def get_songs_artist_ids_playlist(sp, ids):
    playlist = sp.playlist_tracks(ids)
    songs_id = []
    artists_id = []
    for result in playlist['items']:
        songs_id.append(result['track']['id'])
        for artist in result['track']['artists']:
            artists_id.append(artist['id'])
    return songs_id, artists_id


def download_albums(music_id, artist=False):
    if artist == True:
        ids_album = get_albums_id(music_id)
    else:
        if type(music_id) == list:
            ids_album = music_id
        elif type(music_id) == str:
            ids_album = list([music_id])

    tracks = []
    for ids in ids_album:
        # Obtener Ids de canciones en album
        song_ids = get_album_songs_id(ids=ids)
        # Obtener feautres de canciones en album
        ids2 = song_ids

        print(f"Album Length: {len(song_ids)}")

        time.sleep(.6)
        track, columns = get_songs_features(ids2)
        tracks.append(track)

        print(f"Song Added: {track[0]} By {track[2]} from the album {track[1]}")
        #clear_output(wait=True)

    #clear_output(wait=True)
    print("Music Downloaded!")

    return tracks, columns


def download_playlist(id_playlist, n_songs, sp, spt):
    songs_id = []
    tracks = []

    for i in range(0, n_songs, 100):
        playlist = spt.playlist_tracks(id_playlist, limit=100, offset=i)

        for songs in playlist['items']:
            songs_id.append(songs['track']['id'])

    counter = 1
    for ids in songs_id:
        time.sleep(.6)
        track, columns = get_songs_features(sp, ids)
        tracks.append(track)

        print(f"Song {counter} Added:")
        print(f"{track[0]} By {track[2]} from the album {track[1]}")
        counter += 1

    print("Music Features Extracted!")

    return tracks, columns


def main():
    sp, spt = connectSpotifyClient()
    """
    urn = 'spotify:track:0Svkvt5I79wficMFgaqEQJ'
    track = sp.track(urn)
    # pprint(track)

    # ------------------------------------------
    artist_name = 'weezer'
    results = sp.search(q=artist_name, limit=50)
    tids = []
    for i, t in enumerate(results['tracks']['items']):
        print(' ', i, t['name'])
        tids.append(t['uri'])

    start = time.time()
    features = sp.audio_features(tids)
    delta = time.time() - start
    for feature in features:
        print(json.dumps(feature, indent=4))
        print()
        analysis = sp._get(feature['analysis_url'])
        print(json.dumps(analysis, indent=4))
        print()
    print("features retrieved in %.2f seconds" % (delta,))
    """
    tracks, columns = download_playlist('37i9dQZF1DXdPec7aLTmlC', 2, sp, spt)
    # If the id if for artist, you must to put specify True to the artist parameter
    #tracks, columns = download_albums('id_of_the_artist_or_the_album', artist=False)
    df1 = pd.DataFrame(tracks, columns=columns)
    print(df1.head())


if __name__ == "__main__":
    main()