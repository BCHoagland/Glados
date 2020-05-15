import os
import pathlib
import torch
import torch.nn.functional as F

def get_device():
    '''return a reference to the gpu if one's available, else the cpu'''
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def write_file(dirs, filename, text):
    '''
    Write text to the given file in the specified location (overwrites old text in the file if it already exists)

    Inputs
    ------
    dirs: directory path of the new file
    filename: name of the file
    text: what to put in the file
    '''
    pathlib.Path(dirs).mkdir(parents=True, exist_ok=True)
    with open(dirs + '/' + filename, 'w') as f:
        f.write(text)


def save_model(net, filename, epoch_num):
    '''
    Save the given model

    Inputs
    ------
    net: the model to save
    filename: what to name the model
    epoch_num: intended to be the number of epochs completed so far during training, but technically is just a number identifier for the file so that the same model can be saved multiple times at different points during training
    '''
    dirs = f'saved_models/{filename}'
    pathlib.Path(dirs).mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), f'{dirs}/epoch-{epoch_num}')


def read_data(filenames, batch_size, seq_size, val_ratio=0.1):
    '''
    Collect data from given files into a dataset

    Inputs
    ------
    filenames: names of text files that the model will be trained on
    batch_size: number of text sequences included in each training batch
    seq_size: length of each text sequence
    val_ratio: the ratio of the size of the validation set to the size of the whole dataset

    Outputs
    -------
    X: lots of text sequences, split into batches
    Y: same as X, but the text is shifted to the right by one character (we want the model to take in X and output Y)
    X_val, Y_val: validation sets for X and Y
    len(chars): number of unique characters in the training data
    char2int: a dictionary that maps characters to integers
    int2char: a dictionary that maps integers to characters
    num_batches: the number of batches in the dataset
    '''
    # read data from each given file
    text = ''
    if not isinstance(filenames, list): filenames = [filenames]
    for filename in filenames:
        with open(f'data/{filename}', 'r') as f:
            text += f.read()

    # make encoding and decoding dictionaries
    chars = set(text)
    int2char = dict(enumerate(chars))
    char2int = {v: k for k, v in int2char.items()}

    # make data divisible by batch and sequence sizes
    idx = -(len(text) % (batch_size * seq_size))
    encoded = [char2int[c] for c in text][:idx+1]

    # make data into batches
    X = torch.tensor(encoded[:-1]).view(batch_size, -1)
    Y = torch.tensor(encoded[1:]).view(batch_size, -1)

    # ont hot encode the input data
    X = one_hot(X, len(chars))

    # determine where to split data into training and validation data
    idx = int(val_ratio * X.shape[1])
    idx += seq_size - (idx % seq_size)

    # split the data and put it on the right device
    device = get_device()
    X = X[:,:-idx].to(device)
    X_val = X[:,-idx:].to(device)
    Y = Y[:,:-idx].to(device)
    Y_val = Y[:,-idx:].to(device)

    # get number of batches per epoch for training data
    num_batches = X.shape[1] // seq_size

    return X, Y, X_val, Y_val, len(chars), char2int, int2char, num_batches


def one_hot(arr, n_labels):
    '''one-hot encode the text in arr with n_labels possible classes'''
    return F.one_hot(arr, n_labels).float()


def batches(X, Y, batch_size, seq_size):
    '''yield all batches in the dataset, one at a time'''
    num_batches = X.shape[1] // seq_size
    for i in range(num_batches):
        yield X[:, i*seq_size:(i+1)*seq_size], Y[:, i*seq_size:(i+1)*seq_size]