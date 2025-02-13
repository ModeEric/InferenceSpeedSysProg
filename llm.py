import torch
import torch.nn as nn
import regex as re
from collections import defaultdict
from datasets import load_dataset
import json
from tqdm import tqdm
import os
class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_heads, n_blocks, dropout, d_model):
        super().__init__()
        self.block_size = block_size  # Save block_size as instance variable
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, dropout) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Make sure sequence length doesn't exceed block_size
        if T > self.block_size:
            idx = idx[:, :self.block_size]
            targets = targets[:, :self.block_size] if targets is not None else None
            T = self.block_size
            
        # Create position indices on the same device as input
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        tok_embd = self.token_embedding(idx)
        pos_embd = self.position_embedding(pos)
        
        # Add positional embeddings
        x = tok_embd + pos_embd.unsqueeze(0)  # Add batch dimension to pos embeddings
        
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss


class Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model) 
        self.value = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.dropout(y)
        return y


class GPTTokenizer:
    def __init__(self, vocab_size=4096+256+1):
        self.vocab_size = vocab_size
        self.encoder = {'<unk>': 0}  # Initialize with unknown token
        self.decoder = {0: '<unk>'}  # Initialize with unknown token
        self.bpe_ranks = {}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""")
        
    def train(self, texts, min_frequency=2):
        # Count token frequencies
        token_freqs = defaultdict(int)
        for text in texts:
            tokens = self._tokenize(text)
            for token in tokens:
                token_freqs[token] += 1
                
        # Sort tokens by frequency
        sorted_tokens = sorted(token_freqs.items(), key=lambda x: x[1], reverse=True)
        
        # Take only the most frequent tokens up to vocab_size - 1 
        # (subtract 1 because we already have <unk> token)
        vocab_tokens = sorted_tokens[:self.vocab_size - 1]
        
        # Create encoder/decoder
        for token, _ in vocab_tokens:
            if token not in self.encoder:
                idx = len(self.encoder)
                self.encoder[token] = idx
                self.decoder[idx] = token
            
    def _tokenize(self, text):
        """Basic tokenization into words"""
        return self.pat.findall(text)
        
    def _get_pairs(self, vocab):
        """Get counts of adjacent symbol pairs"""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs
        
    def _merge_pair(self, pair, vocab):
        """Merge a symbol pair in all words"""
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in vocab.items():
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = freq
            
        return new_vocab
        
    def encode(self, text):
        """Encode text to token ids"""
        tokens = self._tokenize(text)
        if not tokens:  # Handle empty input
            return [self.encoder['<unk>']]
            
        # Convert tokens to ids, using <unk> for unknown tokens
        ids = []
        for token in tokens:
            token_id = self.encoder.get(token)
            if token_id is None:
                token_id = self.encoder['<unk>']
            ids.append(token_id)
        return ids
        
    def decode(self, ids):
        """Decode token ids back to text"""
        # Convert ids to tokens using decoder, default to <unk> for unknown ids
        tokens = [self.decoder.get(id, '<unk>') for id in ids]
        # Simply join the tokens together
        text = ''.join(tokens)
        return text

def bytes_to_unicode():
    """
    Creates a mapping from bytes to unicode characters for tokenization
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    return dict(zip(bs, cs))

def train_tokenizer(texts, min_frequency=2):
    tokenizer = GPTTokenizer()
    tokenizer.train(texts, min_frequency)
    return tokenizer

if __name__ == "__main__":
    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    documents_train = []
    for i, example in enumerate(ds):
        if i >= 100000:
            break
        documents_train.append(example["text"])

    if not os.path.exists("tokenizer.json"):
        tokenizer = train_tokenizer(documents_train)
    else:
        with open("tokenizer.json", "r") as f:
            encoder = json.load(f)
            vocab_size = len(encoder) + 1  # +1 for <unk> token
            tokenizer = GPTTokenizer(vocab_size=vocab_size)
            tokenizer.encoder = encoder
            tokenizer.decoder = {v: k for k, v in encoder.items()}

    # Verify tokenizer is properly loaded
    if len(tokenizer.encoder) == 0 or len(tokenizer.decoder) == 0:
        print("Warning: Tokenizer vocabulary is empty!")
        tokenizer = train_tokenizer(documents_train)  # Fallback to training new tokenizer

    # Save the tokenizer
    with open("tokenizer.json", "w") as f:
        json.dump(tokenizer.encoder, f)

    # Print vocabulary size for debugging
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Number of tokens in encoder: {len(tokenizer.encoder)}")

    # Train the model with the correct vocabulary size
    model = GPT(vocab_size=tokenizer.vocab_size, block_size=128, n_heads=4, n_blocks=4, dropout=0.0, d_model=128)

    # Debug: Print max token ID
    max_token_id = max(tokenizer.encoder.values())
    print(f"Maximum token ID in encoder: {max_token_id}")

    # count parameters:
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


    for epoch in range(1):
        for i, text in tqdm(enumerate(documents_train)):
            tokens = tokenizer.encode(text)
            if i==0:
                print(tokens)
            if len(tokens) < 2:  # Skip very short sequences
                continue
                
            # Truncate long sequences
            if len(tokens) > 128:  # block_size is 128
                tokens = tokens[:128]
                
            # Verify token IDs are within range
            if max(tokens) >= tokenizer.vocab_size:
                print(f"Warning: Token ID {max(tokens)} >= vocab_size {tokenizer.vocab_size}")
                continue
                
            # Create input and target sequences
            input_tokens = tokens[:-1]
            target_tokens = tokens[1:]
            
            # Convert to tensors and add batch dimension
            input_tokens = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0)
            target_tokens = torch.tensor(target_tokens, dtype=torch.long).unsqueeze(0)
            
            optimizer.zero_grad()
            logits, loss = model(input_tokens, target_tokens)
            loss.backward()
            optimizer.step()
            if i % 1000 == 0:
                print(f"Step {i} loss: {loss.item()}")
        print(f"Epoch {epoch} loss: {loss.item()}")
        
        # Save the model
        torch.save(model.state_dict(), f"model_{epoch}.pt")
