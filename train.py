with open('tiny-shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

'''print(f"Length of dataset in characters: {len(text):,}")'''

'''print(text[:1000])  # Print the first 1000 characters to get a sense of the dataset'''

chars = sorted(list(set(text)))
vocab_size = len(chars)
'''print(f"Unique characters: {''.join(chars)}")
print(f"Vocab size: {vocab_size}")'''

# Create a mapping from characters to integers and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}  # character to integer
itos = {i: ch for i, ch in enumerate(chars)}  # integer to character
def encode(s):
    """Encode a string into a list of integers."""
    return [stoi[c] for c in s]
def decode(l):
    """Decode a list of integers into a string."""
    return ''.join([itos[i] for i in l])

'''print(encode("hii there"))  # Example encoding
print(decode(encode("hii there")))  # Example decoding'''

import torch
data = torch.tensor(encode(text), dtype=torch.long)
'''print(data.shape, data.dtype)  # Print the shape and dtype of the tensor
print(data[:1000])  # Print the first 1000 integers to check encoding'''

# Split the data into train and validation sets
n = int(0.9 * len(data))  # 90% for training, 10% for validation
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size + 1]

x = train_data[:block_size]  # Input sequence
y = train_data[1:block_size + 1]  # Target sequence (next character)
for t in range(block_size):
    context = x[:t + 1]  # Context is the input sequence up to time t
    target = y[t]  # Target is the next character
    '''print(f"when the input is {context}, the target is {target}")'''

torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
'''print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')'''

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        '''print(f"when input is {context.tolist()} the target: {target}")'''

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is a (B,T) tensor of indicies in the current context
        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
             # focus only on the last time step
            logits = logits[:, -1, :]  # (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
        return idx
    
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

print(decode(m.generate(torch.zeros((1, 1), dtype = torch.long), max_new_tokens=100)[0].tolist()))

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for steps in range(10000):

    # sample a batch of data
    xb, yb = get_batch('train')

    # eval the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)  # clear the gradients
    loss.backward()  # backpropagation
    optimizer.step()  # update the parameters

print(loss.item())
print(decode(m.generate(torch.zeros((1, 1), dtype = torch.long), max_new_tokens=400)[0].tolist()))
