import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def downsampling(s_raw, f_s, f_ds=1_000):
    if f_s % f_ds == 0:
        downsample_factor = f_s // f_ds
        s_ds = signal.decimate(s_raw, downsample_factor)
    else:
        nearest_downsample_factor = round(f_s / f_ds)
        new_sample_rate = f_s // nearest_downsample_factor
        
        if new_sample_rate == 0:
            raise ValueError("Der berechnete Downsample-Faktor ist nicht sinnvoll. Überprüfen Sie die Eingabewerte.")

        s_ds = signal.decimate(s_raw, nearest_downsample_factor,ftype='fir')
        print("Not sufficient good implemented yet")
    return s_ds

def bandpass_filter(sig,lowcut, highcut,fs, order):
    sos = signal.butter(order, [lowcut, highcut], btype='band', fs = fs, output = 'sos' )
    bandpass_sig = signal.sosfiltfilt(sos, sig)
    return bandpass_sig


def instantaneous_freq(signal, fs):
    analytic_sig = hilbert(signal)
    inst_phase  = np.unwrap(np.angle(analytic_sig))
    inst_freq = (np.diff(inst_phase)/(2.0*np.pi) * fs)
    return inst_freq