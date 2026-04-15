import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

# defining the dataset class
class LyricsDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=512): # GPT-2 can handle up to 1024 tokens, but 512 keeps memory manageable for my machine
        self.samples = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                song = json.loads(line)
                text = f"<|artist|>{song['artist']}<|lyrics|>\n{song['lyrics']}\n<|end|>"
                tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
                self.samples.append(torch.tensor(tokens))

# __len__ and __getitem__ are required by PyTorch
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
# for the padding
def collate_fn(batch, pad_token_id):
    max_len = max(len(x) for x in batch)
    padded = []
    attention_masks = []

    for tokens in batch:
        padding_length = max_len - len(tokens)
        padded.append(torch.cat([tokens, torch.full((padding_length,), pad_token_id)]))
        attention_masks.append(torch.cat([torch.ones(len(tokens)), torch.zeros(padding_length)]))
        # attention mask tells the model which tokens are real (1) and which are padding (0) so it doesn't try to learn anything from padding tokens
    return {
        "input_ids": torch.stack(padded),
        "attention_mask": torch.stack(attention_masks)
    }

# Training function
def train():
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("train/tokenizer")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer)) # expanding the model's embedding layer because we added new special tokens. this will add randomly initialized embeddings for our new token, the model will learn what these tokens mean during training.

    pad_token_id = tokenizer.pad_token_id

    # Load data
    train_dataset = LyricsDataset("data/lyrics_train.jsonl", tokenizer)
    val_dataset = LyricsDataset("data/lyrics_val.jsonl", tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id)   
    )
        # This is basically equivalent to writing this: 
        # def my_collate(batch):
            # return collate_fn(batch, pad_token_id)
        # train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=my_collate) 
        # Because DataLoader expects a collate_fn that takes one argument (the batch), but our collate_fn needs two arguments

    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id)
)

    # Training setup
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01) # Note: learning rate of 5e-5 is a well-established default for GPT-2 fine-tuning, weight decay of 0.01 is the default value for AdamW
    epochs = 5 # 5 is a reasonable starting point for a small dataset
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps) # during warmup, the learning rate starts near zero. Warming up lets the model gently adjust to the new tokens before training at full intensity. 10% of total steps is a common rule of thumb.

    # Training
    model.train()
    for epoch in range(epochs):
        total_loss = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids) # Note:  this is how causal language model training works. The "label" (correct answer) for each token is the next token.
            loss = outputs.loss

            loss.backward() # backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping (caps them at a maximum magnitude of 1.0)
            optimizer.step() # updates the parameters based on the gradients
            scheduler.step()
            optimizer.zero_grad() #  resets gradients to zero (without this, they'd accumulate across steps)

            total_loss += loss.item()

            if step % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

        # Validation
        model.eval() # for validation, we switch to evaluation mode (disables dropout)
        val_loss = 0
        with torch.no_grad(): # for validation, tells PyTorch not to track gradients (saves memory, speeds things up)
            for batch in val_loader:
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["input_ids"])
                val_loss += outputs.loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} validation loss: {val_loss:.4f}\n")
        model.train()

    # Save the model
    model.save_pretrained("models/gpt2-lyrics")
    tokenizer.save_pretrained("models/gpt2-lyrics")
    print("Model saved to models/gpt2-lyrics/")

if __name__ == "__main__":
    train()