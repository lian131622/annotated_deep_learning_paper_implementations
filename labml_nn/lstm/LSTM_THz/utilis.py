import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import blackmanharris, hamming
import torch
import pywt


def postprocessor(sig):
    # 选择小波基函数
    wavelet = 'db4'

    # 进行离散小波变换
    coeffs = pywt.wavedec(sig, wavelet)

    # 将小波系数置零进行去噪
    threshold = 0.6
    coeffs = [pywt.threshold(c, threshold * max(c)) for c in coeffs]

    # 重构信号
    reconstructed_signal = pywt.waverec(coeffs, wavelet)

    return reconstructed_signal


def hilbert_torch(x, N=None, axis=-1):
    """
    Compute the analytic signal, using the Hilbert transform.

    The transformation is done along the last axis by default.

    Parameters
    ----------
    x : tensor
        Signal data. Must be real.
    N : int, optional
        Number of Fourier components. Default: ``x.shape[axis]``
    axis : int, optional
        Axis along which to do the transformation. Default: -1.

    Returns
    -------
    xa : tensor
        Analytic signal of `x`, of each 1-D array along `axis`

    Notes
    -----
    The analytic signal ``x_a(t)`` of signal ``x(t)`` is:

    .. math:: x_a = F^{-1}(F(x) 2U) = x + i y

    where `F` is the Fourier transform, `U` the unit step function,
    and `y` the Hilbert transform of `x`.

    In other words, the negative half of the frequency spectrum is zeroed
    out, turning the real-valued signal into a complex signal. The Hilbert
    transformed signal can be obtained from ``torch.imag(hilbert_torch(x))``,
    and the original signal from ``torch.real(hilbert_torch(x))``.

    """
    x = torch.as_tensor(x)
    if torch.is_complex(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")

    Xf = torch.fft.fft(x, n=N, dim=axis)
    h = torch.zeros(N, dtype=Xf.dtype, device=Xf.device)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if x.ndim > 1:
        ind = [None] * x.ndim
        ind[axis] = slice(None)
        h = h[tuple(ind)]
    x = torch.fft.ifft(Xf * h, dim=axis)
    return x


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
    filter_function = np.exp(1j * fre_axis * 2 * np.pi * (20)) * (np.cos(2 * np.pi * fre_axis / (4 * cut_off))) ** 2
    cutoff_index = np.where(fre_axis > cut_off)[0]
    if len(cutoff_index) > 0:
        cutoff_index = cutoff_index[0]
        filter_function[cutoff_index:-cutoff_index] = 0
    for irf_fre in irf_fres:
        filtered_values = irf_fre * filter_function
        ans.append(filtered_values)
    plt.figure('Gaussian')
    plt.plot(irf_fre)
    plt.plot(filter_function)
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
