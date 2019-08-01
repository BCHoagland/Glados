import torch

def get_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'

def read_data(filename, batch_size, seq_size):
    # read data
    with open(f'data/{filename}', 'r') as f:
        text = f.read()
    text = text.split()

    # make encoding and decoding dictionaries
    chars = set(text)
    int2char = dict(enumerate(chars))
    char2int = {v: k for k, v in int2char.items()}
    
    # make data divisible by batch and sequence sizes
    idx = -(len(text) % (batch_size * seq_size))
    encoded = [char2int[c] for c in text][:idx+1]

    # make data into batches
    device = get_device()
    X = torch.tensor(encoded[:-1]).view(batch_size, -1).to(device)
    Y = torch.tensor(encoded[1:]).view(batch_size, -1).to(device)

    # get number of batches per epoch
    num_batches = X.shape[1] // seq_size

    return X, Y, len(chars), char2int, int2char, num_batches


def batches(X, Y, batch_size, seq_size):
    num_batches = X.shape[1] // seq_size
    for i in range(num_batches):
        yield X[:, i*seq_size:(i+1)*seq_size], Y[:, i*seq_size:(i+1)*seq_size]