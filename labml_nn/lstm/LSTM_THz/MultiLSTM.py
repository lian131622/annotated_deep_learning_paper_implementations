import argparse
import random

import numpy as np
from labml import lab
from matplotlib import pyplot as plt
from scipy.io import loadmat
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from utilis import add_win, generate_positive_frequency_axis, GaussianFilter_irf_fre, hilbert_torch, postprocessor
from LSTM_LOSS import LSTMModel
from scipy.signal import hilbert


# Function to generate shifted signals
def generate_shifted_signals(signal, shift_amount, num_shifts) -> list:
    shifted_signals = []
    for i in range(num_shifts):
        shifted_signal = np.roll(signal, shift_amount * (i + 1))
        shifted_signals.append(shifted_signal)
    return shifted_signals


# Parameters
N = 4096 * 4
start = 4
end = 300  # the end of the high SNR region
num_shift = 1  # how many shift version is used for training
shift_amount = 1  # how much shift in each delay version
epoch = 200  # epoch for training
time_steps = 20  # how many time steps used for prediction
lr = 0.02  # learning late
pred_future = 200  # how many points to predict
ini_shift = 10  # number of shift to make the beat effect clear

sample_name = '322r1'
index = random.randint(0, 300)
print(index)

# Load data
folder = lab.get_data_path() / 'THzMillScale/Train'
sig_all = loadmat(os.path.join(folder, f'{sample_name}/sig_all.mat'))['sig_all'].T  # all the measurement
ref = loadmat(os.path.join(folder, f'{sample_name}/ref.mat'))['ref'].squeeze()
# select a signal
sig = sig_all[index]


def get_IRFList(sig, ref, s, e):
    """
    shift the signal to get different irf
    Parameters
    ----------
    s: start index for the irf_fre
    e: end index for the irf_fre

    Returns
    -------
    Return a list of IRFs
    """
    sig = np.roll(sig, ini_shift)  # add some roll to have a better indication of envelop
    sig_list = generate_shifted_signals(sig, shift_amount, num_shift)
    irf_List = [(np.fft.fft(add_win(signal), N) / np.fft.fft(add_win(ref), N))[s:e] for signal in sig_list]
    return irf_List


# Prepare data for training
def preprocess_data(irfList, time_steps=50, part='real'):
    all_data = np.array([getattr(irf, part) for irf in irfList])
    X, y = [], []
    for i in range(all_data.shape[0]):
        for j in range(all_data.shape[1] - time_steps):
            X.append(all_data[i, j:j + time_steps])
            y.append(all_data[i, j + time_steps])
    X, y = np.array(X), np.array(y)
    return X.reshape(X.shape[0], X.shape[1], 1), y.reshape(-1, 1)


def get_DataSetandLoader(irf_List, time_Steps):
    """

    Parameters
    ----------
    irf_List
    time_Steps

    Returns
    -------
    Return Dataset and DataLoader
    """
    # Preprocess data
    X_real, y_real = preprocess_data(irf_List, time_Steps, part='real')
    X_imag, y_imag = preprocess_data(irf_List, time_Steps, part='imag')
    # X_all = np.vstack((X_real, X_imag))
    # y_all = np.vstack((y_real, y_imag))
    X_all = X_real
    y_all = y_real
    # Split data into training and testing sets
    split = int(len(X_real) * 1)
    X_train, y_train = X_all[:split], y_all[:split]
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    # Create DataLoader
    trainDataset = TensorDataset(X_train, y_train)
    # Here we don't use shuffle because we add another loss function which relies on the structure of the input
    trainLoader = DataLoader(trainDataset, batch_size=(end - start - time_Steps) * num_shift, shuffle=False)
    return trainDataset, trainLoader


# Define device


# Initialize models, criteria, and optimizers


# Define custom loss function
def envelop_loss(y_pred, num_shift, target):
    y_group = y_pred.reshape(num_shift, -1)
    target = target.reshape(num_shift, -1)
    envelopes_target = []
    envelopes = []
    for row in y_group:
        analytic_signal = hilbert_torch(row)
        amplitude_envelope = torch.abs(analytic_signal)
        envelopes.append(amplitude_envelope.to(y_pred.device))
    for row in target:
        analytic_signal = hilbert_torch(row)
        amplitude_envelope = torch.abs(analytic_signal)
        envelopes_target.append(amplitude_envelope.to(y_pred.device))

    # 将包络从列表转换为tensor
    envelopes = torch.stack(envelopes)
    envelopes_target = torch.stack(envelopes_target)

    envelopes_target_mean = torch.mean(envelopes_target, dim=0)

    # Compute the loss as the mean squared error between envelopes and envelopes_target_mean
    loss = F.mse_loss(envelopes, envelopes_target_mean.expand_as(envelopes))

    return loss, envelopes, envelopes_target_mean


# Save model function
def save_model(model, path):
    torch.save(model.state_dict(), path)


# Load model function
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# Train model function
def train_model(model, train_loader, criterion, optimizer, num_epochs=500, model_path='model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epoch_losses = []
    custom_losses = []
    predict_envelopes = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_custom_loss = 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            y_pred = model(inputs)
            # construct the loss
            # mean_pred = torch.mean(y_pred)
            # mean_loss = mean_pred ** 2
            # envelopLoss, envelopes, target_mean = envelop_loss(y_pred, torch.tensor(num_shift).to(device), targets)
            loss = criterion(y_pred, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # total_custom_loss += envelopLoss.item()

        avg_loss = total_loss / len(train_loader)
        avg_custom_loss = total_custom_loss / len(train_loader)

        # envelopes_mean = torch.mean(envelopes).item()
        # predict_envelopes.append(envelopes_mean)

        epoch_losses.append(avg_loss)
        custom_losses.append(avg_custom_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}, Custom Loss: {avg_custom_loss}')

    save_model(model, model_path)
    print(f'Model saved to {model_path}')

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), epoch_losses, label='Total Loss')
    plt.plot(range(1, num_epochs + 1), custom_losses, label='Custom Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    print('train end')


# Prediction function
def preprocess_new_signal(signal, time_step):
    signal = signal.reshape(-1, 1)
    X = []
    for i in range(len(signal) - time_step):
        X.append(signal[i:i + time_step])
    return np.array(X)


def Predict(model, test_sig, time_step, device, predict_steps=500):
    X_new = preprocess_new_signal(test_sig, time_step)
    X_new = torch.tensor(X_new, dtype=torch.float32).to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for _ in range(predict_steps):
            input_seq = X_new[-1].unsqueeze(0)
            pred = model(input_seq)
            predictions.append(pred.item())
            new_input = np.append(X_new[-1][1:].cpu().numpy(), pred.item()).reshape(-1, 1)
            X_new = torch.cat((X_new, torch.tensor(new_input, dtype=torch.float32).to(device).unsqueeze(0)), dim=0)
    return np.concatenate((test_sig, predictions))


# Main function to parse command-line arguments and execute the script
def main(train_model_flag):
    # get the irfList for a select signal
    irfList = get_IRFList(sig, ref, start, end)
    # Plot the real part of the first IRF
    plt.figure()
    plt.plot(irfList[0].real)
    plt.plot(np.abs(hilbert(irfList[0].real)))
    # get dataset and loader
    train_dataset, train_loader = get_DataSetandLoader(irfList, time_steps)
    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # get model
    model = LSTMModel().to(device)
    optimizer_real = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    if train_model_flag:
        # Train both models for real and imag
        train_model(model, train_loader, criterion, optimizer_real, num_epochs=epoch,
                    model_path=f'model_combine_{sample_name}.pth')
    else:
        # Load trained models for prediction
        model = load_model(LSTMModel().to(device), f'model_combine_{sample_name}.pth')

    # Evaluate on new data
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    new_signal = sig_all[index]
    new_signal = np.roll(new_signal, ini_shift)
    test_irf = (np.fft.fft(add_win(new_signal), N) / np.fft.fft(add_win(ref), N))[:end]
    new_irf_ref = (np.fft.fft(add_win(new_signal), N) / np.fft.fft(add_win(ref), N))
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    predictions_real = Predict(model, test_irf.real, time_steps, device, pred_future)
    predictions_imag = Predict(model, test_irf.imag, time_steps, device, pred_future)

    plt.figure()
    plt.subplot(211)
    plt.plot(predictions_real, label='Predicted Real')
    plt.plot(new_irf_ref.real, label='Actual Signal Real')
    plt.plot(test_irf.real, label='Test Signal Real')
    plt.xlim((0, 400))
    plt.ylim((-5, 5))
    plt.legend()
    plt.subplot(212)
    plt.plot(predictions_imag, label='Predicted Imag')
    plt.plot(new_irf_ref.imag, label='Actual Signal Imag')
    plt.plot(test_irf.imag, label='Test Signal Imag')
    plt.xlim((0, 400))
    plt.ylim((-5, 5))
    plt.legend()

    # Create symmetrical signal
    def create_symmetrical_signal(signal, target_length, type):
        if len(signal) < target_length:
            padding_value = signal[-1]
            padded_signal = np.pad(signal, (0, target_length - len(signal)), 'constant',
                                   constant_values=0)
        else:
            padded_signal = signal
        flipped_signal = padded_signal[::-1]
        if type == 'real':
            symmetrical_signal = np.concatenate((padded_signal, flipped_signal))
        elif type == 'imag':
            symmetrical_signal = np.concatenate((padded_signal, -flipped_signal))
        else:
            print('define the type of prediction')
            return None
        return symmetrical_signal

    predictions_real = create_symmetrical_signal(predictions_real, N // 2, 'real')
    predictions_imag = create_symmetrical_signal(predictions_imag, N // 2, 'imag')

    new_irf = predictions_real + 1j * predictions_imag

    new_irf_filter = GaussianFilter_irf_fre(generate_positive_frequency_axis(0.0083, N), [new_irf],
                                            1 / 0.0083 / N * (end + pred_future))
    # new_irf_filter = [new_irf]

    plt.figure()
    plt.plot(new_irf.real)
    plt.plot(new_irf_filter[0].real, alpha=0.5)

    plt.figure()
    plt.plot(new_irf.imag)
    plt.plot(new_irf_filter[0].imag, alpha=0.5)

    plt.figure()
    plt.plot(np.real(np.fft.ifft(new_irf_filter[0])), label='Time domain irf')
    plt.legend()
    plt.grid()

    waveletSignal = postprocessor(np.real(np.fft.ifft(new_irf_filter[0])))
    plt.figure()
    plt.plot(waveletSignal)
    plt.grid()
    plt.show(block=True)
    print('End')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or load models for IRF prediction.')
    parser.add_argument('--train', action='store_true', help='Train the models if this flag is set.')
    args = parser.parse_args()
    main(args.train)
