with open('tiny-shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

'''print(f"Length of dataset in characters: {len(text):,}")'''

#print(text[:1000])  # Print the first 1000 characters to get a sense of the dataset

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
    print(f"when the input is {context}, the target is {target}")

