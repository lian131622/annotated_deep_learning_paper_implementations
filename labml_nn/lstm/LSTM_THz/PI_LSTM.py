from lstm import LSTMFeatureExtractor, CNN1D, SimpleNN
import torch.nn as nn


class PI_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, n_lstm, sig_stp, samplingPeriod, N, cut_off,
                 num_filters, kernel_size):
        super().__init__()

        # Initialize a list of LSTM feature extractors
        self.lstm_feature_extractors = LSTMFeatureExtractor(input_size, hidden_size, num_layers, num_filters, kernel_size)

        # 1d convolution method
        self.cnn1d = CNN1D(100, 65)

        # simple dense network
        self.simpleNN = SimpleNN(40, 65)

        # Define the classification layer
        self.classifier = nn.Linear(65, num_classes)

    def forward(self, sig):
        features = self.simpleNN(sig)
        # features = sig.unsqueeze(1)
        # features = self.cnn1d(sig)
        # features = features.unsqueeze(-1)
        # features = self.lstm_feature_extractors(features)
        output = self.classifier(features)
        return output
