import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import OrderedDict
import numpy as np

def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()

class Policy(nn.Module):
    def __init__(self, input_size, output_size, max_action, device='cpu'):
        super(Policy, self).__init__()
        self.max_action = max_action
        self.device = device
        self.l1 = nn.Linear(input_size, 128).to(device)
        self.l2 = nn.Linear(128, 128).to(device)
        self.l3 = nn.Linear(128, output_size).to(device)
        #self.apply(weight_init)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x

    def count_parameters(self):
        count = 0
        for param in self.parameters():
            count += param.cpu().data.numpy().flatten().shape[0]
        return count 

    def es_params(self):
        return [(k, v) for k, v in zip(self.state_dict().keys(),
            self.state_dict().values())]


class LSTMPolicy(nn.Module):
    def __init__(self, input_size, output_size, max_action, device='cpu'):
        super(LSTMPolicy, self).__init__()
        self.device = device
        self.max_action = max_action
        self.embed_size = 64
        self.hidden_size = 64
        self.embed = nn.Linear(input_size, self.embed_size).to(device)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size).to(device)
        self.linear = nn.Linear(self.hidden_size, output_size).to(device)
        #self.apply(weight_init)
        self.init_hidden()

    def init_hidden(self):
        ### (num_layers*num_directions, batch_size, hidden_size)
        h0 = torch.from_numpy(np.zeros((1, 1, self.hidden_size))).float().to(
                self.device)
        c0 = torch.from_numpy(np.zeros((1, 1, self.hidden_size))).float().to(
                self.device)
        self.hidden = (h0, c0)


    def forward(self, x):
        self.init_hidden()
        ### input: (seq_len, batch_size, dim)
        ### embedding: (seq_len, batch_size, embed_size)
        embedding = F.relu(self.embed(x))

        ### lstm_out: (seq_len, batch_size, hidden_size)
        lstm_out, (ht, ct) = self.lstm(embedding, self.hidden)

        ### out: (batch_size, output_size)
        out = self.max_action * torch.tanh(self.linear(F.relu(lstm_out[-1])))
        return out

    def count_parameters(self):
        count = 0
        for param in self.parameters():
            count += param.cpu().data.numpy().flatten().shape[0]
        return count 

    def es_params(self):
        return [(k, v) for k, v in zip(self.state_dict().keys(),
            self.state_dict().values())]

