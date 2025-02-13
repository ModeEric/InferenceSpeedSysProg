#Test model_9.pth

import torch
from llm import GPT, GPTTokenizer
import json

# Load the tokenizer from saved file first
with open("tokenizer.json", "r") as f:
    encoder = json.load(f)
    vocab_size = 4354  # Set fixed size to match model
    tokenizer = GPTTokenizer(vocab_size=vocab_size)
    tokenizer.encoder = encoder
    tokenizer.decoder = {v: k for k, v in encoder.items()}

# Add some debug prints
print("Vocabulary size:", len(encoder))
print("Sample from encoder:", list(encoder.items())[:5])

# Initialize model with the correct vocab size (4354)
model = GPT(vocab_size=4354, block_size=128, n_heads=4, n_blocks=4, dropout=0.0, d_model=128)
model.load_state_dict(torch.load("model_0.pt"))
model.eval()  # Set to evaluation mode

# Test model on sample text
text = "Hello, how are you?"
# Debug the encoding process
tokens = tokenizer.encode(text)
print("Tokens:", tokens)
print("Token meanings:", [tokenizer.decoder.get(t, '<unk>') for t in tokens])

# Tokenize text
encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long).unsqueeze(0)  # Add batch dimension
print("Encoded:", encoded)

# Test model
with torch.no_grad():
    logits, loss = model(encoded)
    # Get predictions
    predictions = torch.argmax(logits, dim=-1)
    decoded = tokenizer.decode(predictions[0].tolist())
    print("Model output:", decoded)

# Print original text for comparison
print("Original text:", text)

