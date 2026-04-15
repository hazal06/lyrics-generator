from transformers import GPT2Tokenizer

def setup_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Add our special tokens
    special_tokens = {
        "additional_special_tokens": ["<|artist|>", "<|lyrics|>", "<|end|>"], # <|end|> serves a similar purpose to GPT-2's built-in <|endoftext|> but is specific to our song format
        "pad_token": "<|pad|>" # GPT-2 doesn't come with a pad token by default, so we add one
    }
    tokenizer.add_special_tokens(special_tokens)

    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    # print(f"Additional special tokens: {special_tokens['additional_special_tokens']}")
    print(f"Pad token: {tokenizer.pad_token}")

    # Save for reuse
    tokenizer.save_pretrained("train/tokenizer")
    print("Tokenizer saved to train/tokenizer/")

    return tokenizer

if __name__ == "__main__":
    tokenizer = setup_tokenizer()

    # Demo: see how a training sample gets tokenized
    sample = "<|artist|>Taylor Swift<|lyrics|>\nI walked through the door\n<|end|>"
    tokens = tokenizer.encode(sample)
    print(f"\nSample text:\n{sample}")
    print(f"\nToken IDs: {tokens}")
    print(f"\nDecoded back: {tokenizer.decode(tokens)}")