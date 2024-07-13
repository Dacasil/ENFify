import numpy as np
from utils import sym_phase_interval


def classic_phase_analysis(s_tone, f_s, N_DFT):
    M = len(s_tone)
    hann_window = np.hanning(M)
    x = s_tone * hann_window
    X = np.fft.fft(x, n=N_DFT)

    k_peak = np.argmax(np.abs(X[: N_DFT // 2]))
    f_DFT = k_peak * f_s / N_DFT

    phi_DFT = np.angle(X[k_peak])
    return f_DFT, phi_DFT


def secure_precise_phase_analysis(s_tone, f_s, N_DFT, phi_DFT=None, timeout_iterations=1_00):
    """wrapper fucntion for precise_phase_analysis. In case it doesn't work, it retries while increasing N_DFT"""
    for i in range(timeout_iterations):
        try:
            f_DFT1, phi_DFT1 = precise_phase_analysis(s_tone, f_s, N_DFT + i, phi_DFT)
        except ValueError:
            pass
        else:
            return f_DFT1, phi_DFT1
    raise TimeoutError(
        f"precise_phase_estimation could not converge in {timeout_iterations} iterations"
    )


def precise_phase_analysis(s_tone, f_s, N_DFT, phi_DFT=None):
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
    phi_DFT1 = np.arctan(numerator / denominator)
    # phi_DFT1 = np.arctan2(numerator, denominator)

    # choose value closer to classic phase analysis
    # if phi_DFT is None:
    #     _, phi_DFT = classic_phase_analysis(s_tone, f_s, N_DFT)

    # # choosing the phi_DFT1 closer to phi_DFT
    # _phi_DFT1 = sym_phase_interval(phi_DFT1 + np.pi)
    # if np.abs(_phi_DFT1 - phi_DFT) < np.abs(phi_DFT1 - phi_DFT):
    #     phi_DFT1 = _phi_DFT1

    return f_DFT1, phi_DFT1
