import numpy as np
from enfify.enf_estimation import phase_estimation_DFT1, phase_estimation_DFT0


def test_phase_estimation_DFT1():
    sample_freq = 1_000
    enf_freq = 50.2
    phi_0 = np.pi / 2
    M = 200
    time = np.arange(M) / sample_freq
    enf_component = np.cos(2 * np.pi * enf_freq * time + phi_0)

    NDFT = 2_000
    f0_estimated, phi0_estimated = phase_estimation_DFT0(enf_component, sample_freq, NDFT)

    phase_estimation = phase_estimation_DFT1(
        enf_component,
        sample_freq,
        NDFT,
        f0_estimated,
    )

    print(f"True phase: {phi_0}")
    print(f"Estimated phase: {phase_estimation}")


if __name__ == "__main__":
    test_phase_estimation_DFT1()
