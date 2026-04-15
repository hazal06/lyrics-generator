import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_lyrics(artist, prompt="", max_length=300, temperature=0.8):
    tokenizer = GPT2Tokenizer.from_pretrained("models/gpt2-lyrics")
    model = GPT2LMHeadModel.from_pretrained("models/gpt2-lyrics")
    model.eval()

    # Build the input
    input_text = f"<|artist|>{artist}<|lyrics|>\n"
    if prompt:
        input_text += prompt + "\n"

    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.encode("<|end|>")[0]
        )

    text = tokenizer.decode(output[0], skip_special_tokens=False)

    # Clean up: extract just the lyrics part
    if "<|lyrics|>" in text:
        text = text.split("<|lyrics|>")[1]
    if "<|end|>" in text:
        text = text.split("<|end|>")[0]

    return text.strip()

if __name__ == "__main__":
    artists = ["Taylor Swift", "Pink Floyd", "Kendrick Lamar"]

    for artist in artists:
        print(f"\n{'='*50}")
        print(f"  {artist}")
        print(f"{'='*50}")
        print(generate_lyrics(artist))
        print()