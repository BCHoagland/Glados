import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, tokens, n_hidden=256, n_layers=2, drop_prob=0.5):
        super().__init__()

        self.chars = tokens
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        # network layers
        self.lstm1 = nn.LSTM(len(self.chars), n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout1 = nn.Dropout(drop_prob)
        self.lstm2 = nn.LSTM(n_hidden, n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout2 = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, len(self.chars))


    def forward(self, x, hidden):
        out, hidden = self.lstm1(x, hidden)
        out = self.dropout1(out)
        out, hidden = self.lstm2(out, hidden)
        out = self.dropout2(out)
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)
        return out, hidden


    def init_hidden(self, batch_size, train_on_gpu=False):
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden
