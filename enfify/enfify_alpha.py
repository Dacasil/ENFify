import os
#import argparse
import numpy as np
import typer
import yaml
from enf_enhancement import VariationalModeDecomposition
from enf_estimation import (
    segmented_phase_estimation_DFT0,
    segmented_phase_estimation_hilbert,
)
from preprocessing import bandpass_filter, downsampling
from visualization import (
    create_cut_phase_plot,
    create_phase_plot,
    cut_to_alpha_pdf,
    to_alpha_pdf,
)
from Rodriguez_Audio_Authenticity import find_cut_in_phases
from utils import add_defaults, read_wavfile

# CONSTANTS
app = typer.Typer()


# FUNCTIONS
@app.command()
def frontend(
    audio_file_path: str = typer.Argument(
        "INPUT_Audio_Data/cut_min_001_ref.wav",  # TODO: am ende kein default
        help="The path of the audio file to process.",
    ),
    config_path: str = typer.Option(
        "config.yml",  # TODO: am ende kein default
        help="The path of the configuration file to use.",
    ),
):
    """
    ENFify - Audio Tampering Detection Tool

    Args:
        audio_file_path: The path of the audio file to process.
        config_file_path: The path to the config file.
    """

    print(f"Processing audio file: {audio_file_path}")

    # Load config and defaults
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    with open("defaults.yml", "r") as f:
        defaults = yaml.safe_load(f)
    add_defaults(config, defaults)

    # Read data
    sig, fs = read_wavfile(audio_file_path)

    main(sig, fs, config)


def main(sig, fs, config):
    # ENF EXTRACTION

    # Downsampling
    downsample_config = config["downsample"]
    if downsample_config["is_enabled"]:
        f_ds = downsample_config["downsampling_frequency"]

        sig, fs = downsampling(sig, fs, f_ds)

    # Bandpass Filter
    bandpass_config = config["bandpassfilter"]
    if bandpass_config["is_enabled"]:
        lowcut = bandpass_config["lowcut"]
        highcut = bandpass_config["highcut"]
        sig = bandpass_filter(sig, lowcut, highcut, fs, 4)

    # Variational Mode Decomposition
    vmd_config = config["VMD"]
    if vmd_config["is_enabled"]:
        alpha = vmd_config["alpha"]
        tau = vmd_config["tau"]
        n_mode = vmd_config["n_mode"]
        DC = vmd_config["DC"]
        tol = vmd_config["tol"]

        u_clean, _, _ = VariationalModeDecomposition(sig, alpha, tau, n_mode, DC, tol)
        sig = u_clean[0]

    # ENF ANALYSIS

    # Phase Estimation
    nom_enf = config["nominal_enf"]

    phase_config = config["phase_estimation"]
    num_cycles = phase_config["num_cycles"]
    n_dft = phase_config["n_dft"]

    time = len(sig) / fs

    # hilbert phase estimation
    hilbert_phases = segmented_phase_estimation_hilbert(sig, fs, num_cycles, nom_enf)
    x_hilbert = np.linspace(0.0, time, len(hilbert_phases))

    # DFT0 instantaneous phase estimation
    phases = segmented_phase_estimation_DFT0(sig, fs, num_cycles, n_dft, nom_enf)
    x_DFT0 = np.linspace(0.0, time, len(phases))

    hilbert_phases_new, x_hilbert_new, hil_interest_region = find_cut_in_phases(
        hilbert_phases, x_hilbert
    )
    DFT0_phases_new, x_DFT0_new, DFT0_interest_region = find_cut_in_phases(phases, x_DFT0)

    # Create the phase plots
    # TODO: Paths in config or as terminal arguments
    hilbert_phase_path = "temp/hilbert_phase_im.png"
    DFT0_phase_path = "temp/DFT0_phase_im.png"
    pdf_outpath = "temp/enfify_alpha.pdf"
    os.makedirs("temp", exist_ok=True)


    if np.any(hil_interest_region) == False:
        create_phase_plot(x_hilbert, hilbert_phases, hilbert_phase_path)
        create_phase_plot(x_DFT0, phases, DFT0_phase_path)
        to_alpha_pdf(hilbert_phase_path, DFT0_phase_path, pdf_outpath)

    else:
        create_cut_phase_plot(
            x_hilbert_new,
            hilbert_phases_new,
            x_hilbert,
            hil_interest_region,
            hilbert_phase_path,
        )
        create_cut_phase_plot(
            x_DFT0_new,
            DFT0_phases_new,
            x_DFT0,
            DFT0_interest_region,
            DFT0_phase_path,
        )

        cut_to_alpha_pdf(hilbert_phase_path, DFT0_phase_path, pdf_outpath)



# MAIN
if __name__ == "__main__":
    app()
# else:
#     from rodriguez import generate_single_tone

#     config_path = "config.yml"
#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f) or {}
#     with open("defaults.yml", "r") as f:
#         defaults = yaml.safe_load(f)
#     add_defaults(config, defaults)

#     if True:  # synthetic
#         fs = int(8e3)
#         sig = generate_single_tone(50.001, fs)
#         # TODO: Add cut
#     else:  # file
#         sig, fs = read_wavfile(
#             "/home/leo/enfify/data/scratch/silva_data/INPUT_Audio_Data/cut_min_001_ref.wav"
#         )

#     main(sig, fs, config)
