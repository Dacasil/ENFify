import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

sys.path.insert(0, '/home/leo_dacasi/Dokumente/summerofcode/ENFify/enfify')

from enf_estimation import segmented_freq_estimation_DFT1, segmented_freq_estimation_hilbert
from preprocessing import bandpass_filter
from enf_enhancement import RFA, VariationalModeDecomposition, stft_search

import soundfile as sf
import math
import cmath

noise_fs, down_sig_noise = wavfile.read('/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/interim/ENF-WHU-Dataset/1min_noise/1min_noisy_down/down_min_001.wav')
ref_fs, down_sig_ref = wavfile.read('/home/leo_dacasi/Dokumente/summerofcode/Enfify Data Synced/interim/ENF-WHU-Dataset/1min_ref/1min_enf_ref_data/min_001_ref.wav')

n_noise = np.arange(len(down_sig_noise))
n_ref = np.arange(len(down_sig_ref))

lowcut = 49.5
highcut = 50.5
bandpass_sig = bandpass_filter(down_sig_noise, lowcut, highcut, noise_fs, 1)
bandpass_ref = bandpass_filter(down_sig_ref, lowcut, highcut, ref_fs, 1)

'''
freqs_DFT1 = segmented_freq_estimation_DFT1(bandpass_sig, noise_fs, 100, 20_000, 50)
freqs_ref = segmented_freq_estimation_DFT1(bandpass_ref, ref_fs, 100, 20_000, 50)

plt.figure(figsize=(10, 4))
plt.plot(freqs_DFT1, color="red")
plt.plot(freqs_ref, color="orange")
plt.title('Downsampled noisy tone')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid()
plt.show()
'''

fs = noise_fs
f0 = 50
I = 7
epsilon = 1e-20
tau = int(750)

denoised_signal = RFA(bandpass_sig, noise_fs, tau, epsilon, I, f0)
