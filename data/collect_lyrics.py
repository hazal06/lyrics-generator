import os
import json
import time
from dotenv import load_dotenv
import lyricsgenius

load_dotenv()
token = os.getenv("GENIUS_API_TOKEN") # get the Genius Lyrics Client Access Token

genius = lyricsgenius.Genius(token, timeout=30, retries=8, sleep_time=1) # to avoid timeout error
genius.remove_section_headers = False  # keep [Chorus], [Verse] etc for now
genius.skip_non_songs = True           # skip track lists, liner notes
genius.excluded_terms = ["(Remix)", "(Live)"]  # skip remixes and live versions

artists = ["Taylor Swift", "Pink Floyd", "Kendrick Lamar"]

def collect_artist_lyrics(artist_name, max_songs=200):
    print(f"Fetching songs for {artist_name}...")
    try:
        artist = genius.search_artist(artist_name, max_songs=max_songs, sort="popularity")
    except Exception as e:
        print(f"Error fetching {artist_name}: {e}")
        return []
    
    if artist is None:
        print(f"Could not find {artist_name}")
        return []
    
    songs = []
    for song in artist.songs:
        lyrics = song.lyrics
        if lyrics and len(lyrics) > 100:  # skip very short entries
            songs.append({
                "artist": artist_name,
                "title": song.title,
                "lyrics": lyrics
            })
    
    print(f"Collected {len(songs)} songs for {artist_name}")
    return songs

if __name__ == "__main__":
    all_songs = []

    for artist in artists:
        songs = collect_artist_lyrics(artist, max_songs=200)
        all_songs.extend(songs)
        time.sleep(30)  # pause between artists to be polite to the API

    output_path = "data/lyrics_raw.jsonl"
    with open(output_path, "a", encoding="utf-8") as f: # append ("a") instead of overwrite ("o")
        for song in all_songs:
            f.write(json.dumps(song, ensure_ascii=False) + "\n")

    print(f"\nDone! Saved {len(all_songs)} songs to {output_path}")