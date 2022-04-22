import spotify_connect as spotify
def connect_playlist():
    sp, spt = spotify.connectSpotifyClient()
    playlist_link = "https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC"
    playlist_URI = playlist_link.split("/")[-1].split("?")[0]
    #print(sp.playlist_tracks('37i9dQZF1DXdPec7aLTmlC', fields=None, limit=100, offset=0, market=None))
    offset_init=0
    track_uris=[]
    while offset_init <= 500:
        track_uris.append([x["track"]["uri"] for x in sp.playlist_tracks('1llkez7kiZtBeOw5UjFlJq', fields=None, limit=100, offset=offset_init, market=None)["items"]])
        offset_init +=100
        print(track_uris)
    print(track_uris)




if __name__ == "__main__":
    connect_playlist()