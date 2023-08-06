# Griffin-Lim implementation by candlewill
# Audio -> Spectrogram / Spectrogram -> Audio conversion
# https://github.com/candlewill/Griffin_lim
from matplotlib import pyplot as plt
import librosa
import librosa.filters
import numpy as np
import pickle
from scipy import signal
import soundfile as sf
import random
import os

num_mels = 64  # 128
num_freq = 256  # 1013 #128
sample_rate = 8000  # 16000
frame_length_ms = 40.0  # 32.0
frame_shift_ms = 20  # 16.0
preemphasis = 0.97
min_level_db = -80
ref_level_db = 20
griffin_lim_iters = 300  # 60


def load_wav(path, sr):
    return librosa.load(path, sr=sr)[0]


def save_wav(wav, path):
    sf.write(path, wav, sample_rate, subtype='PCM_16')


def spectrogram(y):
    D = _stft(_preemphasis(y))
    S = _amp_to_db(np.abs(D)) - ref_level_db
    return _normalize(S)


def inv_spectrogram(spectrogram):
    S = _db_to_amp(_denormalize(spectrogram) + ref_level_db)  # Convert back to linear
    return _inv_preemphasis(_griffin_lim(S ** 1.5))  # Reconstruct phase


def melspectrogram(y):
    D = _stft(_preemphasis(y))
    S = _amp_to_db(_linear_to_mel(np.abs(D)))
    return _normalize(S)


def inv_melspectrogram(melspectrogram):
    S = _mel_to_linear(_db_to_amp(_denormalize(melspectrogram)))  # Convert back to linear
    return _inv_preemphasis(_griffin_lim(S ** 1.5))  # Reconstruct phase


# Based on https://github.com/librosa/librosa/issues/434
def _griffin_lim(S):
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex_)
    for i in range(griffin_lim_iters):
        if i > 0:
            angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y


def _stft(y):
    n_fft = (num_freq - 1) * 2
    hop_length = int(frame_shift_ms / 1000. * sample_rate)
    win_length = int(frame_length_ms / 1000. * sample_rate)
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _istft(y):
    hop_length = int(frame_shift_ms / 1000. * sample_rate)
    win_length = int(frame_length_ms / 1000. * sample_rate)
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


# Conversions:
_mel_basis = None
_inv_mel_basis = None


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _mel_to_linear(mel_spectrogram):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis())
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))


def _build_mel_basis():
    n_fft = (num_freq - 1) * 2
    return librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=num_mels)


def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _preemphasis(x):
    return signal.lfilter([1, -preemphasis], [1], x)


def _inv_preemphasis(x):
    return signal.lfilter([1], [1, -preemphasis], x)


def _normalize(S):
    return np.clip((S - min_level_db) / -float(min_level_db), 0, 1)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db


def _plot_time(x):
    time = np.linspace(0, len(x) / sample_rate, num=len(x))
    fig, ax = plt.subplots()
    ax.plot(time, x)
    ax.set(xlabel='Time (sec)', ylabel='Amplitude', title='Time domain plot')
    # fig.savefig("time.png")
    plt.show()


def _plot_spectrum(x, title, fs=sample_rate):
    n = len(x)  # length of the signal
    df = fs / n  # frequency increment (width of freq bin)
    # compute Fourier transform, its magnitude and normalize it before plotting
    Xfreq = np.fft.fft(x)
    XMag = abs(Xfreq) / n
    # Note: because x is real, we keep only the positive half of the spectrum
    # Note also: half of the energy is in the negative half (not plotted)
    XMag = XMag[0:int(n / 2)]
    # freq vector up to Nyquist freq (half of the sample rate)
    freq = np.arange(0, fs / 2, df)
    fig, ax = plt.subplots()
    ax.plot(freq, XMag)
    ax.set(xlabel='Frequency (Hz)', ylabel='Magnitude', title=f'Frequency domain plot of {title}')
    # fig.savefig("freq.png")
    plt.show()


def _mask_gen(w, h, gap_ratio_min=0.2, gap_ratio_max=0.5):
    mask = np.ones((h, w))
    gap_len_min = int(gap_ratio_min * w)
    gap_len_max = int(gap_ratio_max * w)
    gap_num = random.randint(1, 8)
    total_gap_len = random.randint(gap_len_min, gap_len_max)
    while (gap_num > 0) and (total_gap_len > 0):
        center = random.randint(40, w - 40)
        gap = random.randint(20, total_gap_len)
        min_gap_time = np.max([center - (gap // 2), 0])
        max_gap_time = np.min([(center + (gap // 2)), w])
        mask[:, min_gap_time:max_gap_time] = 0
        gap_num += -1
        total_gap_len = total_gap_len - (max_gap_time - min_gap_time)  # MAHSA: changed total_gap_len-gap
        if total_gap_len < 50:
            break

    return mask


def save_environmental_sound():
    aug_wav1 = load_wav(os.path.join('/Users/kadkhodm/Desktop/Research/inpainting/datasets/noise',
                                     'airplane-landing_daniel_simion_8k.wav'), sr=sample_rate)
    aug_mel_sgram1 = melspectrogram(aug_wav1[:23824]).astype(np.float32)
    aug_wav2 = load_wav(os.path.join('/Users/kadkhodm/Desktop/Research/inpainting/datasets/noise',
                                     'heavy-rain-daniel_simon_8k.wav'), sr=sample_rate)
    aug_mel_sgram2 = melspectrogram(aug_wav2[:23824]).astype(np.float32)
    aug_wav3 = load_wav(os.path.join('/Users/kadkhodm/Desktop/Research/inpainting/datasets/noise',
                                     'labrador-barking-daniel_simon_8k.wav'), sr=sample_rate)
    aug_mel_sgram3 = melspectrogram(aug_wav3[:23824]).astype(np.float32)
    aug_wav4 = load_wav(os.path.join('/Users/kadkhodm/Desktop/Research/inpainting/datasets/noise',
                                     'old-car-engine_daniel_simion_8k.wav'), sr=sample_rate)
    aug_mel_sgram4 = melspectrogram(aug_wav4[:23824]).astype(np.float32)

    aug_mel_list = [aug_mel_sgram1, aug_mel_sgram2, aug_mel_sgram3, aug_mel_sgram4]
    with open('/Users/kadkhodm/Desktop/Research/inpainting/datasets/cache_noise.cache', 'wb') as fp:
        pickle.dump(aug_mel_list, fp)


def load_environmental_sound():
    # with open('/Users/kadkhodm/Desktop/Research/inpainting/datasets/cache_noise.cache', 'rb') as fp:
    #     aug_mel_sgram_list = pickle.load(fp)
    coin = np.random.uniform()
    print("coin: ", coin)
    if coin < 0.25:
        aug_wav1 = load_wav(os.path.join('/Users/kadkhodm/Desktop/Research/inpainting/datasets/noise',
                                         'airplane-landing_daniel_simion_8k.wav'), sr=sample_rate)
        aug_mel_sgram = melspectrogram(0.05 * aug_wav1[:23824]).astype(np.float32)
    elif 0.25 <= coin < 0.5:
        aug_wav2 = load_wav(os.path.join('/Users/kadkhodm/Desktop/Research/inpainting/datasets/noise',
                                         'heavy-rain-daniel_simon_8k.wav'), sr=sample_rate)
        aug_mel_sgram = melspectrogram(0.05 * aug_wav2[:23824]).astype(np.float32)
    elif 0.5 <= coin < 0.75:
        aug_wav3 = load_wav(os.path.join('/Users/kadkhodm/Desktop/Research/inpainting/datasets/noise',
                                         'labrador-barking-daniel_simon_8k.wav'), sr=sample_rate)
        aug_mel_sgram = melspectrogram(0.05 * aug_wav3[:23824]).astype(np.float32)
    else:
        aug_wav4 = load_wav(os.path.join('/Users/kadkhodm/Desktop/Research/inpainting/datasets/noise',
                                         'old-car-engine_daniel_simion_8k.wav'), sr=sample_rate)
        aug_mel_sgram = melspectrogram(0.05 * aug_wav4[:23824]).astype(np.float32)

    return aug_mel_sgram


def get_intrusions_mask(frame_dim, spec_len, cov_mean, cov_std, n_max_intr, min_intr_len=3):
    # number of intrusions selection
    n_intr = random.randint(1, n_max_intr)
    # mask coverage selection
    mask_cov = max(min_intr_len * n_intr / spec_len, min(random.gauss(cov_mean, cov_std), 0.8))
    mask_bins = int(np.around(spec_len * mask_cov))
    true_mask_cov = mask_bins / spec_len  # it is slightly different from sampled value due to rounding
    # distribution of mask to intrusions
    intr_lens = []
    for i in range(0, n_intr):
        if i == n_intr - 1:
            intr_lens.append(mask_bins - sum(intr_lens))
        elif i == 0:
            intr_lens.append(random.randint(min_intr_len, max(min_intr_len,
                                                              int((mask_bins - min_intr_len * (
                                                                      n_intr - 1)) * np.exp(
                                                                  -(n_intr - 1) / 6)))))
        else:
            intr_lens.append(random.randint(min_intr_len, max(min_intr_len, int((mask_bins - sum(
                intr_lens) - min_intr_len * (n_intr - i - 1)) * np.exp(-(n_intr - 1) / 6)))))
    random.shuffle(intr_lens)

    # intrusions onset selection
    onset_pos = []
    for i, l in enumerate(intr_lens):
        if i == 0 and i == n_intr - 1:
            onset_pos.append(random.randint(0, spec_len - mask_bins))
        elif i == 0:
            onset_pos.append(random.randint(0, (spec_len - mask_bins - (n_intr - 1))) // 2)
        elif i == n_intr - 1:
            onset_pos.append(
                random.randint(onset_pos[-1], onset_pos[-1] + intr_lens[i - 1] + 1 + spec_len - intr_lens[i]))
        else:
            onset_pos.append(random.randint(onset_pos[-1] + intr_lens[i - 1] + 1, (
                    onset_pos[-1] + intr_lens[i - 1] + 1 + spec_len - sum(intr_lens[i:]) - (n_intr - i - 1)) // 2))

    # create mask
    mask = np.ones([spec_len, frame_dim])
    for os, il in zip(onset_pos, intr_lens):
        mask[os: os + il] = 0

    return mask, true_mask_cov, n_intr


def pmsq_stft(x, w, step):
    # Short-time Fourier Transform
    #   x    : input signal vector
    #   w    : vector with the analysis window to be used
    #   step : window shift
    # Zero padding to STFT calculation
    nsampl = len(x)
    wlen = len(w)
    nframe = int(np.ceil((float(nsampl - wlen) / step))) + 1
    dif = wlen + (nframe - 1) * step - nsampl
    x = np.append(x, np.zeros(dif))
    # Zero padding in the edges
    ntotal = nsampl + dif + 2 * (wlen - step)
    x = np.append(np.zeros(wlen - step), np.append(x, np.zeros(wlen - step)))
    # DFT computation per frame (hal spectra)
    Xtf = np.array([np.fft.rfft(w * x[i:i + wlen])
                    for i in range(0, ntotal - wlen + 1, step)]) + 1e-12 * (1 + 1j)

    return Xtf


def pmsqe_istft(Xtf, w, step, nsampl):
    # Inverse Short-time Fourier Transform (aka Overlap and Add)
    #   Xtf  : input matrix with STFT
    #   w    : vector with the window used during analysis
    #   step : window shift

    # Parameters
    nframe, nbin = Xtf.shape
    wlen = len(w)
    ntotal = (nframe - 1) * step + wlen

    # Overlapp-add method
    x = np.zeros(ntotal)

    ind = 0
    for i in range(0, ntotal - wlen + 1, step):
        Xt = Xtf[ind]
        x[i:i + wlen] += w * np.fft.irfft(Xt)
        ind += 1

    return x[(wlen - step):(wlen - step) + nsampl]
