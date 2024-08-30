import torch
import torch.nn as nn


class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_filters, kernel_size):
        super(LSTMFeatureExtractor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Extract features from the last hidden state
        features = out[:, -1, :]
        return features


class CNN1D(nn.Module):
    def __init__(self, feature_channel, N):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * (N // 8), feature_channel)  # Adjust based on sequence length

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        # x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the output
        x = torch.relu(self.fc1(x))
        return x


class SimpleNN(nn.Module):
    def __init__(self, input_dim, middle_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, middle_dim)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        return x
