import random
from scipy.signal import medfilt
from labml import lab, tracker, experiment, monit
import torch.utils.data
import os
from scipy.io import loadmat
import numpy as np
from scipy.signal import windows as wd
from scipy.signal.windows import blackmanharris, hamming
from labml.configs import BaseConfigs, option
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from labml_helpers.device import DeviceConfigs
from PI_LSTM import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class Configs(BaseConfigs):
    device: torch.device = DeviceConfigs()

    sampling_period = 0.0083
    N = 4096
    # epochs
    epoches = 1000
    # the channel of the input signal
    input_size = 1
    # hidden for lstm
    hidden_size = 2
    # layer number for lstm
    num_layers = 1
    # number of classes
    n_classes = 3
    # number of lstm feature extractor
    n_lstm = 1
    # the cutoff frequency for filtering
    cut_off = 2.5
    # signal shift for sig
    sig_step = 50
    # batch size
    batch_size = 128
    # learning rate
    learning_rate: float = 0.08
    # Dataset
    dataset: torch.utils.data.Dataset
    # Dataloader
    data_loader: torch.utils.data.DataLoader
    # Adam optimizer
    optimizer: torch.optim.Adam
    pi_lstm: PI_LSTM

    def init(self):
        self.pi_lstm = PI_LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.n_classes,
            n_lstm=self.n_lstm,
            sig_stp=self.sig_step,
            samplingPeriod=self.sampling_period,
            N=self.N,
            cut_off=self.cut_off,
            num_filters=32,
            kernel_size=3
        ).to(self.device)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, self.batch_size, shuffle=True)
        self.optimizer = torch.optim.Adam(self.pi_lstm.parameters(), lr=self.learning_rate)

    def train(self):
        for sigs, labels in self.data_loader:
            sigs = sigs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.pi_lstm(sigs)
            criterion = nn.CrossEntropyLoss()
            labels = labels.long()
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            accuracy = 100 * correct / total

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            tracker.save('loss', loss)
            tracker.save('accuracy', accuracy)  # 记录准确率


    def run(self):
        for _ in monit.loop(self.epoches):
            self.train()
            tracker.new_line()
            experiment.save_checkpoint()


def create_window(length, sym):
    return hamming(length, sym=sym)


def add_win(Signal, sym=False):
    if len(Signal) == 0:
        raise ValueError("Signal is empty")

    if not sym:
        sample_peak = np.argmax(Signal)
        win_len = len(Signal) - sample_peak
        win_right = create_window(2 * win_len, sym=False)
        win_left = create_window(2 * sample_peak, sym=False)

        win = np.concatenate((win_left[:sample_peak], win_right[win_len:]))
    else:
        win = create_window(len(Signal), sym=True)

    return win * Signal


def shift_signal(ref, sig, shift_distance):
    ref_max_idx = np.argmax(ref)
    sig_max_idx = np.argmax(sig)
    shift_amount = (ref_max_idx + shift_distance) - sig_max_idx
    return np.roll(sig, shift_amount)


def GaussianFilter_irf_fre(fre_axis, irf_fres, cut_off=4):
    ans = []
    cutoff_frequency = cut_off  # 示例值
    sigma = cutoff_frequency / np.sqrt(2 * np.log(2))  # 高斯标准差
    filter_function = np.exp(1j * fre_axis * 2 * np.pi * (10)) * (np.cos(2 * np.pi * fre_axis / (4 * cut_off))) ** 2
    cutoff_index = np.where(fre_axis > cut_off)[0]
    if len(cutoff_index) > 0:
        cutoff_index = cutoff_index[0]
        filter_function[cutoff_index:-cutoff_index] = 3E-12
    for irf_fre in irf_fres:
        filtered_values = irf_fre * filter_function
        ans.append(filtered_values)
    return ans


def DoubleGaussian(N, irf_fres):
    shift_peak = 15
    time_axis = np.arange(0, 4096 * 0.0083, 0.0083)
    lf, hf = 0.0014, 0.2
    f_DG_t = hf * np.exp(-(time_axis - shift_peak) ** 2 / hf ** 2) - lf * np.exp(
        -(time_axis - shift_peak) ** 2 / lf ** 2)

    f_DG_fft = np.fft.fft(f_DG_t, n=4 * N)
    ans = []
    for irf_fre in irf_fres:
        filtered_values = irf_fre * f_DG_fft
        ans.append(filtered_values)
    return ans


def generate_positive_frequency_axis(sampling_period, signal_length):
    frequency_resolution = 1 / (signal_length * sampling_period)

    frequencies = np.fft.fftfreq(signal_length, d=sampling_period)

    return frequencies


class MillScaleDataset(torch.utils.data.Dataset):
    def __init__(self, samplingperiod, N, pca_components=40):
        super().__init__()

        self._thickDic = {
            '319r2': 14.57,
            '319r1': 14.57,
            '320r1': 5.43,
            '320r2': 5.43,
            '320f1': 5.94,
            '320f2': 5.94,
            '317f1': 6.43,
            '317f2': 6.43,
            '315r1': 6.75,
            '315r2': 6.75,
            '317r1': 7.5,
            '317r2': 7.5,
            '315f1': 7.89,
            '315f2': 7.89,
            '319f1': 8.47,
            '319f2': 8.47,
            '318r1': 8.85,
            '318r2': 8.85,
            '1r1': 9.28,
            '3r1': 9.38,
            '4f1': 9.49,
            '4f2': 9.49,
            '318f1': 10.17,
            '318f2': 10.17,
            '4r1': 10.86,
            '4r2': 10.86,
            '312f1': 11.22,
            '312f2': 11.22,
            '1f1': 11.32,
            '1f2': 11.32,
            '11f1': 12.18,
            '11f2': 12.18,
            '322r1': 12.5,
            '322r2': 12.5,
            '9r1': 12.58,
            '9r2': 12.58,
            '11r1': 12.69,
            '11r2': 12.69,
            '11r3': 12.69,
            '9f1': 12.89,
            '9f2': 12.89,
            '322f1': 13.18,
            '322f2': 13.18
        }

        folder = lab.get_data_path() / 'THzMillScale/Train'
        self._sampleNames = []
        self._labels = []
        self._irf = []
        self._input = []

        # get all the folder names
        for item in os.listdir(folder):
            if os.path.isdir(os.path.join(folder, item)):
                if item not in self._thickDic.keys():
                    continue
                self._sampleNames.append(item)

        # ger reference signal
        ref = loadmat(os.path.join(folder, '319r1/ref.mat'))['ref'].squeeze()

        # prepare the data
        for item in self._sampleNames:
            # for each sample get related signal
            sig_all = loadmat(os.path.join(folder, item, 'sig_all.mat'))['sig_all'].T  # get all the signals\
            # get raw irf in frequency domain
            irf = [(np.fft.fft(add_win(sig), 4 * N) / np.fft.fft(add_win(ref), 4 * N)).real for sig in sig_all]
            # get spectrum for the irf in frequency domain
            # spectrum = [np.abs(np.fft.fft(sig))[:160] for sig in sig_all]
            # # get filtered irf in frequency
            # irf = GaussianFilter_irf_fre(generate_positive_frequency_axis(samplingperiod, 4 * N), irf, 4)

            irf = DoubleGaussian(N, irf)

            # # turn into time domain
            irf = [np.fft.ifft(i).real[:4096] for i in irf]
            # # put the irf to the return variable
            self._input.extend(irf)

            # set the label
            if self._thickDic[item] < 8:
                self.label = 0
            elif 8 <= self._thickDic[item] < 12:
                self.label = 1
            elif 12 < self._thickDic[item]:
                self.label = 2
            self._labels = self._labels + [self.label] * sig_all.shape[0]

        # do PCA
        self._input = np.array(self._input)  # turn back to ndarray
        # scaler = StandardScaler()
        # self._input = scaler.fit_transform(self._input)

        # 应用PCA降维
        if pca_components:
            pca = PCA(n_components=pca_components)
            self._input = pca.fit_transform(self._input)

        # Turn to tensor
        self._labels = torch.tensor(self._labels, dtype=torch.int)
        self._input = [torch.tensor(i, dtype=torch.float32) for i in self._input]

    def __len__(self):
        return len(self._input)

    def __getitem__(self, idx):
        return self._input[idx], self._labels[idx]


@option(Configs.dataset, 'THZMillScale')
def THzMillScale_dataset(c: Configs):
    return MillScaleDataset(c.sampling_period, c.N)


def main():
    experiment.create(name='MillScale', writers={'screen', 'labml'})
    configs = Configs()

    experiment.configs(configs, {
        'dataset': 'THZMillScale',
        'sampling_period': 0.0083,
        'N': 4096,
    })

    configs.init()

    # Set models for saving and loading
    experiment.add_pytorch_models({'pi_lstm': configs.pi_lstm})

    # Start and run the training loop
    with experiment.start():
        configs.run()

    print('end')


#  used to check what input is given to the model
def Test():
    dataset = MillScaleDataset(0.0083, 4096)

    def plot_signals(dataset, num_samples=5):
        # Find indices of each label
        label_indices = {0: [], 1: [], 2: []}
        for idx, label in enumerate(dataset._labels):
            label_indices[label.item()].append(idx)

        fig, ax = plt.subplots(num_samples, 1, figsize=(15, 5))

        for i in range(num_samples):
            for label in label_indices.keys():
                sample_idx = random.choice(label_indices[label])
                sample, ref, _ = dataset[sample_idx]
                ax[i].plot(sample.numpy(), label=f'Label {label} Sample {sample_idx}')

            ax[i].legend()
            ax[i].set_title(f'Sample {i + 1}')

        plt.tight_layout()
        plt.show()

    plot_signals(dataset)
    print('end')


if __name__ == '__main__':
    main()
