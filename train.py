import numpy as np
import torch
import torch.nn.functional as F

from model import RNN
from utils import read_data, batches, get_device
from visualize import progress, plot


batch_size = 128
seq_size = 100
vis_iter = 100
device = get_device()

X, Y, n_chars, char2int, int2char, num_batches = read_data('vonnegut', batch_size, seq_size)

net = RNN(n_chars).to(device)
opt = torch.optim.Adam(net.parameters(), lr=0.001)

'''
TRAINING
'''
def train(epochs=20):
    for epoch in range(epochs):

        batch_num = 0
        h = net.blank_hidden(batch_size)
        for x, y in batches(X, Y, batch_size, seq_size):
            out, h = net(x, h)
            h = tuple([state.detach() for state in h])

            loss = F.cross_entropy(out.transpose(1,2), y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if num_batches * epoch + batch_num % vis_iter == vis_iter - 1:
                plot(num_batches * epoch + batch_num, loss, 'Loss', 'Training', '#FA5784')
            progress(batch_num, num_batches, num_batches * epoch + batch_num, epochs * num_batches)                                     # FIX

            batch_num += 1


'''
GENERATION
'''
def next_char(x):
    probs, indices = torch.topk(x, k=5)
    probs, indices = probs.squeeze().tolist(), indices.squeeze().tolist()
    probs = [p / sum(probs) for p in probs]
    choice = int2char[np.random.choice(indices, p=probs)]
    return choice

def generate():
    with torch.no_grad():
        first_chars = list('I will')
        chars = ''.join(first_chars)

        # run initial chars through model to generate hidden states
        h = net.blank_hidden()
        for c in first_chars:
            x, h = net(torch.tensor([[char2int[c]]]).to(device), h)

        # generate new chars
        for _ in range(100):
            choice = next_char(x)
            chars += choice
            x, h = net(torch.tensor([[char2int[choice]]]).to(device), h)

        print(chars)

'''
"DO IT" - Palpatine
'''
train(epochs=100)
generate()