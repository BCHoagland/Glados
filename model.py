import torch
import torch.nn as nn

from visualize import box
from utils import get_device

class RNN(nn.Module):
    def __init__(self, n_chars, n_hidden=512, n_layers=2, drop_prob=0.5):
        super().__init__()
        
        # params
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.device = get_device()

        # network layers
        self.embed = nn.Embedding(n_chars, n_hidden)
        self.lstm = nn.LSTM(n_hidden, n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.linear = nn.Linear(n_hidden, n_chars)

        # box('Network Architecture')
        # print(self)


    def forward(self, x, hidden):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        out = self.linear(out)
        return out, hidden


    def blank_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.n_hidden).to(self.device),
                torch.zeros(self.n_layers, batch_size, self.n_hidden).to(self.device))