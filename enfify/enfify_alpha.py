import numpy as np
import os
import typer
import yaml
from ENF_Enhancement import VariationalModeDecomposition
from PDF_and_Plot import (
    cut_to_alpha_pdf,
    to_alpha_pdf,
    read_wavfile,
    create_phase_plot,
    create_cut_phase_plot,
)
from ENF_preprocessing import downsampling, bandpass_filter
from ENF_frequency_phase_estimation import (
    segmented_phase_estimation_DFT0,
    segmented_phase_estimation_hilbert,
)
from Rodriguez_Audio_Authenticity import find_cut_in_phases


# CONSTANTS
app = typer.Typer()

# Load defaults
with open("defaults.yml", "r") as f:
    DEFAULTS = yaml.safe_load(f)


# FUNCTIONS
@app.command()
def process_audio(
    audio_file_path: str = typer.Argument(
        "/home/leo/enfify/data/scratch/silva_data/INPUT_Audio_Data/cut_min_001_ref.wav",
        help="The path of the audio file to process.",
    ),
    config_path: str = typer.Option(
        "config.yml", help="The path of the configuration file to use."
    ),
):
    """
    ENFify - Audio Tampering Detection Tool

    Args:
    audio_file_path: The path of the audio file to process.
    config_file_path: The path to the config file.
    """

    print(f"Processing audio file: {audio_file_path}")

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    # add defaults
    add_defaults(config, DEFAULTS)

    # Read data
    sig, fs = read_wavfile(audio_file_path)

    # Data Preprocessing
    sig, fs = data_preprocessing(sig, fs, audio_file_path, config)

    # Phase Analysis
    analyze_phase(sig, fs, config)


def data_preprocessing(sig, fs, audio_file_path, config):
    # Downsampling
    if config["downsample"]:
        downsample_freq = config["downsampling_freq"]
        output_file = os.path.join(
            os.path.dirname(audio_file_path),
            "downsampled_" + os.path.basename(audio_file_path),
        )
        downsampling(audio_file_path, output_file, downsample_freq)
        sig, fs = read_wavfile(output_file)
        os.remove(output_file)

    # Bandpass Filter
    if config["bandpassfilter"]:
        lowcut = config["bandpass_lowcut"]
        highcut = config["bandpass_highcut"]
        sig = bandpass_filter(sig, lowcut, highcut, fs, 4)

    # Variational Mode Decomposition
    if config["VMD"]:
        vmd_params = config["VMD_params"]
        alpha = vmd_params["alpha"]
        tau = vmd_params["tau"]
        n_mode = vmd_params["n_mode"]
        DC = vmd_params["DC"]
        tol = vmd_params["tol"]

        u_clean, _, _ = VariationalModeDecomposition(sig, alpha, tau, n_mode, DC, tol)
        sig = u_clean[0]

    return sig, fs


def analyze_phase(sig, fs, config):
    # TODO: Schalter in Config, welche Teile der Analye durchgeführt werden sollen

    nom_enf = config["expected_enf"]
    NUM_CYCLES = config["num_cycles"]
    N_DFT = config["n_dft"]
    time = len(sig) / fs

    # Hilbert instantaneous phase estimation
    hilbert_phases = segmented_phase_estimation_hilbert(sig, fs, NUM_CYCLES, nom_enf)
    x_hilbert = np.linspace(0.0, time, len(hilbert_phases))

    # DFT0 instantaneous phase estimation
    DFT0_phases = segmented_phase_estimation_DFT0(sig, fs, NUM_CYCLES, N_DFT, nom_enf)
    x_DFT0 = np.linspace(0.0, time, len(DFT0_phases))

    hilbert_phases_new, x_hilbert_new, hil_interest_region = find_cut_in_phases(
        hilbert_phases, x_hilbert
    )
    DFT0_phases_new, x_DFT0_new, DFT0_interest_region = find_cut_in_phases(DFT0_phases, x_DFT0)

    # Create the phase plots
    # TODO: Paths in config
    hilbert_phase_path = "temp/hilbert_phase_im.png"
    DFT0_phase_path = "temp/DFT0_phase_im.png"
    pdf_outpath = "temp/enfify_alpha.pdf"
    os.makedirs("temp", exist_ok=True)

    if hil_interest_region == 0:
        create_phase_plot(x_hilbert, hilbert_phases, hilbert_phase_path)
        create_phase_plot(x_DFT0, DFT0_phases, DFT0_phase_path)
        to_alpha_pdf(hilbert_phase_path, DFT0_phase_path)

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


def add_defaults(config, defaults):
    """
    Recursively add default values to a config dictionary.

    Args:
        config (dict): The configuration dictionary to update.
        defaults (dict): The dictionary containing default values.
    """
    for key, value in defaults.items():
        if key not in config or config[key] is None:
            config[key] = value
        elif isinstance(value, dict) and isinstance(config.get(key), dict):
            add_defaults(config[key], value)


# MAIN
if __name__ == "__main__":
    app()
