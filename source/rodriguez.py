from math import ceil
import numpy as np
import scipy.signal as signal
from utils import sym_phase_interval


# III


def phase_estimation_DFT0(s_tone, f_s, N_DFT):
    M = len(s_tone)
    hann_window = np.hanning(M)
    x = s_tone * hann_window
    X = np.fft.fft(x, n=N_DFT)

    k_peak = np.argmax(np.abs(X[: N_DFT // 2]))
    f_DFT = k_peak * f_s / N_DFT

    phi_DFT = np.angle(X[k_peak])
    return f_DFT, phi_DFT


def repeated_phase_DFT1(s_tone, f_s, N_DFT, phi_DFT=None, timeout_iterations=1_00):
    """wrapper fucntion for precise_phase_analysis. In case it doesn't work, it retries while increasing N_DFT"""
    for i in range(timeout_iterations):
        try:
            f_DFT1, phi_DFT1 = phase_estimation_DFT1(s_tone, f_s, N_DFT + i, phi_DFT)
        except ValueError:
            pass
        else:
            return f_DFT1, phi_DFT1
    raise TimeoutError(
        f"precise_phase_estimation could not converge in {timeout_iterations} iterations"
    )


def phase_estimation_DFT1(s_tone, f_s, N_DFT, phi_DFT=None):
    M = len(s_tone)

    # 1
    s_dash = f_s * np.diff(s_tone)
    s_tone = s_tone[1:]

    # 2
    hann_window = np.hanning(M - 1)
    x = s_tone * hann_window
    x_dash = s_dash * hann_window

    # 3
    X = np.fft.fft(x, n=N_DFT)
    X_dash = np.fft.fft(x_dash, n=N_DFT)

    # 4
    abs_X = np.abs(X)
    abs_X_dash = np.abs(X_dash)
    k_peak = np.argmax(abs_X)

    # 5
    F_k_peak = (np.pi * k_peak) / (N_DFT * np.sin(np.pi * k_peak / N_DFT))

    DFT_0_peak = abs_X[k_peak]
    DFT_1_peak = F_k_peak * abs_X_dash[k_peak]

    # 6
    f_DFT1 = 1 / (2 * np.pi) * DFT_1_peak / DFT_0_peak
    k_DFT1 = N_DFT * f_DFT1 / f_s

    # validating
    if not -0.5 <= k_DFT1 - k_peak < 0.5:
        raise ValueError(f"invalid result: k_peak: {k_peak}, k_DFT1: {k_DFT1}")

    omega_0 = 2 * np.pi * f_DFT1 / f_s
    k_low = np.floor(k_DFT1).astype("int")
    k_high = np.ceil(k_DFT1).astype("int")
    theta_low = np.angle(X_dash[k_low])
    theta_high = np.angle(X_dash[k_high])

    theta = np.interp(k_DFT1, [k_low, k_high], [theta_low, theta_high])

    numerator = np.tan(theta) * (1 - np.cos(omega_0)) + np.sin(omega_0)
    denominator = 1 - np.cos(omega_0) - np.tan(theta) * np.sin(omega_0)
    # phi_DFT1 = np.arctan(numerator / denominator)
    phi_DFT1 = np.arctan2(numerator, denominator)

    # choose value closer to classic phase analysis
    # if phi_DFT is None:
    #     _, phi_DFT = classic_phase_analysis(s_tone, f_s, N_DFT)

    # # choosing the phi_DFT1 closer to phi_DFT
    # _phi_DFT1 = sym_phase_interval(phi_DFT1 + np.pi)
    # if np.abs(_phi_DFT1 - phi_DFT) < np.abs(phi_DFT1 - phi_DFT):
    #     phi_DFT1 = _phi_DFT1

    return f_DFT1, phi_DFT1


# IV A


def generate_single_tone(f_tone, f_s, phi_0=0, amplitude=1, duration=20):
    M = duration * f_s

    n = np.arange(M)
    s_raw = amplitude * np.cos(2 * np.pi * f_tone / f_s * n + phi_0)
    return s_raw


def insert_cut(s_in, f_s, nominal_enf=50, nominal_cycles=1.25, location=0.5):
    # insert cut of ceil 1.25 nominal cycles
    cutlen = ceil(nominal_cycles * f_s / nominal_enf)

    i = round(len(s_in) * location)
    mask = np.ones_like(s_in, dtype=bool)
    mask[i : i + cutlen] = False
    s_in = s_in[mask]
    return s_in


def downsampling(s_raw, f_s, f_ds=1_000):
    if f_s % f_ds == 0:
        downsample_factor = f_s // f_ds
        s_ds = signal.decimate(s_raw, downsample_factor)
    else:
        raise NotImplementedError("Not implemented yet")
        # s_ds = signal.resample_poly(audio_signal, up=3, down=20)
    return s_ds


def fir_filter(s_in, f_s, center_freq=50, passband_width=1.4, num_taps=10_000):
    band = [center_freq - passband_width / 2, center_freq + passband_width / 2]
    filter_taps = signal.firwin(num_taps, band, pass_zero=False, fs=f_s)
    s_out = signal.filtfilt(filter_taps, [1.0], s_in)

    # delete samples at start and end
    s_out[-num_taps // 2 :] = np.nan
    s_out[: num_taps // 2] = np.nan

    return s_out


def segmented_phase_estimation(s_in, f_s, num_cycles, N_DFT, nominal_enf):
    step_size = f_s // nominal_enf  # samples per nominal enf cycle

    num_blocks = len(s_in) // step_size - (num_cycles - 1)

    segments = [s_in[i * step_size : (i + num_cycles) * step_size] for i in range(num_blocks)]

    phases = []
    for segment in segments:
        _, phase = phase_estimation_DFT0(segment, f_s, N_DFT)
        phases.append(phase)

    phases = np.array(phases)
    phases = np.unwrap(phases)
    return phases
