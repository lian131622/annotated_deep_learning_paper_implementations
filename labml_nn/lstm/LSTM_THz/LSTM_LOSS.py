import numpy as np
from labml import lab, tracker, experiment, monit
from matplotlib import pyplot as plt
from scipy.io import loadmat
import os
from utilis import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=128, output_size=1, num_layers=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_layer_size)
        self.linear1 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        norm_out = self.layer_norm(lstm_out[:, -1, :])
        predic = self.tanh(self.linear1(norm_out))
        predic = self.linear2(predic)
        return predic
