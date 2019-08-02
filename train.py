import sys
import torch
import torch.nn.functional as F

from model import RNN
from utils import read_data, one_hot, batches, get_device
from visualize import progress, plot


batch_size = 128
seq_size = 100
vis_iter = 20
grad_norm = 5
device = get_device()

# generate dataset
try: filename = sys.argv[1]
except: filename = 'shakespeare'
X, Y, n_chars, char2int, int2char, num_batches = read_data(filename, batch_size, seq_size)

# make network and optimizer
net = RNN(n_chars).to(device)
opt = torch.optim.Adam(net.parameters(), lr=0.005)

'''
TRAINING
'''
def train(epochs=20):
    iters = 0
    for epoch in range(epochs):

        batch_num = 0
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

            # print progress occasionally
            if iters % vis_iter == vis_iter - 1:
                plot(iters, loss, 'Loss', 'Training', '#FA5784')
            progress(batch_num, num_batches, iters, epochs * num_batches, epoch)

            batch_num += 1
            iters += 1


'''
GENERATION
'''
# convert network output to character
def net2char(x, top_k=5):
    # get top k probabilities
    probs = F.softmax(x.squeeze(), dim=0)
    probs, choices = torch.topk(probs, k=top_k)

    # sample from the top k choices
    idx = torch.multinomial(probs, 1)
    return int2char[choices[idx].item()]

# take a single character and encode it so it works as network input
def format_input(x):
    x = torch.tensor([[char2int[x]]])
    return one_hot(x, n_chars).to(device)

# generate multiple example text chunks
def generate(first_chars='A', example_len=100, examples=1):
    with torch.no_grad():
        for _ in range(examples):
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
            print('-' * 40 + f'\n{chars}')


'''
"DO IT" - Palpatine
'''
train(epochs=30)
generate(example_len=1000, examples=4)