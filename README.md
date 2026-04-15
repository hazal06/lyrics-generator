# Lyrics Generator

A fine-tuned GPT-2 model that generates song lyrics conditioned on artist style. Give it an artist name and it writes new lyrics that reflect their tone, vocabulary, and structure.

Trained on Taylor Swift, Pink Floyd, and Kendrick Lamar for now, three artists with different styles, to test whether a small language model can learn and distinguish between them.

## How It Works

The model uses special tokens to condition on artist style like this:

<|artist|>Taylor Swift<|lyrics|>
We were both young when I first saw you...
<|end|>

At generation time, you provide the artist tag and the model continues in that style.

## Architecture

- **Base model:** GPT-2 (124M parameters)
- **Method:** Full fine-tune with custom tokenizer (4 added special tokens)
- **Training data:** ~200 songs across 3 artists for now, scraped from Genius
- **Training:** 5 epochs, batch size 4, AdamW optimizer (lr 5e-5), linear warmup schedule

## Results

| Metric | Value |
|--------|-------|
| Final training loss | TBD |
| Final validation loss | TBD |
| Style classifier accuracy | TBD |
| Distinct-1 | TBD |
| Distinct-2 | TBD |


## Quickstart

```bash
# Clone and install
git clone https://github.com/hazal06/lyrics-generator.git
cd lyrics-generator
pip install -r requirements.txt

# You'll need a Genius API token for data collection
# Get one at https://genius.com/api-clients
echo "GENIUS_API_TOKEN=your_token" > .env

# Collect data and train
python data/collect_lyrics.py
python data/preprocess.py
python train/train.py

# Generate lyrics
python generate.py
```

## Project Structure

├── data/
│   ├── collect_lyrics.py    # Genius API scraping
│   └── preprocess.py        # Cleaning and train/val split
├── train/
│   ├── tokenizer_setup.py   # Custom tokenizer with special tokens
│   └── train.py             # Fine-tuning loop
├── eval/
│   ├── evaluate.py          # Perplexity and distinctness metrics (coming soon)
│   └── style_classifier.py  # Artist style accuracy (coming soon)
├── generate.py              # Lyrics generation script
└── app.py                   # Gradio web demo (maybe coming soon)

## Next Steps

- [ ] Train LoRA adapter on Mistral-7B for higher quality output
- [ ] Build style classifier to measure artist accuracy
- [ ] Add Gradio web demo with artist blending
- [ ] Deploy to HuggingFace Spaces

## Limitations

This is a learning project. The generated lyrics are original model output, not reproductions of copyrighted songs. The training data was collected via the Genius API for educational purposes only. Song lyrics are copyrighted by their respective owners.