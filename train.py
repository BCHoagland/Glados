import sys
import torch
import torch.nn.functional as F

from model import RNN
from utils import read_data, one_hot, batches, get_device, write_file, save_model
from visualize import progress, plot


# hyperparameters
batch_size = 128
seq_size = 100                      # length of text sequences used during training
save_iter = 50                      # how often the model gets saved during training
grad_norm = 5                       # max gradient component (all gradient components greater than this will be clipped)
device = get_device()               # gpu if one's available, cpu if not

# generate dataset
files = sys.argv[1:] if len(sys.argv) > 1 else 'shakespeare'
filename = '-'.join(files)
X, Y, X_val, Y_val, n_chars, char2int, int2char, num_batches = read_data(files, batch_size, seq_size)

# make network and optimizer
net = RNN(n_chars).to(device)
opt = torch.optim.Adam(net.parameters(), lr=0.005)


############
# TRAINING #
############

def validation_loss():
    '''Return the model loss on the validation set'''
    with torch.no_grad():
        val_losses = []
        val_h = net.blank_hidden(batch_size)
        for x, y in batches(X_val, Y_val, batch_size, seq_size):
            out_val, val_h = net(x, val_h)
            val_loss = F.cross_entropy(out_val.transpose(1,2), y)
            val_losses.append(val_loss)
        val_losses = torch.stack(val_losses)
    return val_losses


def train(epochs=20):
    '''
    Train the RNN and save the model

    Inputs
    ------
    epochs: the number of rounds we train on the whole dataset
    '''
    iters = 0
    for epoch in range(epochs):

        batch_num = 0
        losses = []
        h = net.blank_hidden(batch_size)
        for x, y in batches(X, Y, batch_size, seq_size):

            # use network predictions to compute loss
            h = tuple([state.detach() for state in h])
            out, h = net(x, h)
            loss = F.cross_entropy(out.transpose(1,2), y)

            # optimization step
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_norm)
            opt.step()

            # print training progress
            progress(batch_num, num_batches, iters, epochs * num_batches, epoch + 1)

            # bookkeeping
            losses.append(loss)
            batch_num += 1
            iters += 1

        # plot loss after every epoch
        plot(epoch + 1, torch.stack(losses), 'Loss', 'Training', '#5DE58D', refresh=False)
        plot(epoch + 1, validation_loss(), 'Loss', 'Validation', '#4AD2FF')

        # save the model occasionally
        if epoch % save_iter == save_iter - 1:
            save_model(net, filename, epoch + 1)
    
    # save model at the end of training
    save_model(net, filename, 'final')


##############
# GENERATION #
##############

# 
def net2char(x, top_k=5):
    '''
    convert network output to a single character
    
    Inputs
    ------
    x: output from network
    top_k: characters are chosen stochastically. To speed up computation, we only consider the k most probable characters

    Outputs
    -------
    a single character
    '''
    # get top k probabilities
    probs = F.softmax(x.squeeze(), dim=0)
    probs, choices = torch.topk(probs, k=top_k)

    # sample from the top k choices
    idx = torch.multinomial(probs, 1)
    return int2char[choices[idx].item()]


def format_input(x):
    '''take a single character x and encode it so it works as network input'''
    x = torch.tensor([[char2int[x]]])
    return one_hot(x, n_chars).to(device)


def generate(first_chars='A', example_len=100):
    '''
    Generate example text chunk

    Inputs
    ------
    first_chars: the first characters of the generated text
    example_len: number of characters in the generated text (excluding first_chars)
    '''
    with torch.no_grad():
        first_chars = list(first_chars)
        chars = ''.join(first_chars)

        # run initial chars through model to generate hidden states
        h = net.blank_hidden()
        for c in first_chars:
            inp = format_input(c)
            x, h = net(inp, h)

        # generate new chars
        for _ in range(example_len):
            choice = net2char(x)
            chars += choice

            inp = format_input(choice)
            x, h = net(inp, h)

        # print the results
        return f'\n{chars}'


#######################
# "DO IT" - Palpatine #
#######################
train(epochs=100)

print('Generating text...', end='', flush=True)
text = generate(example_len=10000)
write_file('results', filename, text)
print('DONE')
