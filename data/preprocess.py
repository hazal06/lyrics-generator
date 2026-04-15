import json
import re
import random

random.seed(42)  # makes the train/val split reproducible

def clean_lyrics(lyrics, title):
    # Remove the title line that Genius prepends
    lines = lyrics.split("\n")
    if title.lower() in lines[0].lower():
        lines = lines[1:]
    
    # Remove "You might also like" Genius ads
    text = "\n".join(lines)
    text = re.sub(r"You might also like", "", text)
    
    # Remove contributor/embed text (e.g., "123 ContributorsTranslations...")
    text = re.sub(r"\d*Contributors.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"Embed$", "", text, flags=re.MULTILINE)
    
    # Remove section headers like [Verse 1], [Chorus]
    headers = re.findall(r"\[.*?\]", text)  # save them before removing
    text_no_headers = re.sub(r"\[.*?\]", "", text)
    
    # Clean up extra blank lines
    text_no_headers = re.sub(r"\n{3,}", "\n\n", text_no_headers).strip()
    
    return text_no_headers, headers

if __name__ == "__main__":
    # Load raw data
    with open("data/lyrics_raw.jsonl", "r", encoding="utf-8") as f:
        songs = [json.loads(line) for line in f]

    # Clean each song
    cleaned = []
    for song in songs:
        lyrics, headers = clean_lyrics(song["lyrics"], song["title"])
        if len(lyrics) > 50:  # skip if almost nothing left after cleaning
            cleaned.append({
                "artist": song["artist"],
                "title": song["title"],
                "lyrics": lyrics,
                "sections": headers
            })

    print(f"Kept {len(cleaned)} songs after cleaning (from {len(songs)} raw)")

    # Shuffle and split 90/10, stratified by artist
    by_artist = {}
    for song in cleaned:
        by_artist.setdefault(song["artist"], []).append(song)

    train, val = [], []
    for artist, songs_list in by_artist.items():
        random.shuffle(songs_list)
        split_idx = max(1, int(len(songs_list) * 0.9))
        train.extend(songs_list[:split_idx])
        val.extend(songs_list[split_idx:])
        print(f"  {artist}: {split_idx} train, {len(songs_list) - split_idx} val")

    # Save
    for name, data in [("train", train), ("val", val)]:
        path = f"data/lyrics_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for song in data:
                f.write(json.dumps(song, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(train)} train, {len(val)} val")