import numpy as np
import os
import yaml
import typer
from enfify.enf_enhancement import VMD, RFA
from enfify.enf_estimation import segmented_phase_estimation_hilbert
from enfify.preprocessing import bandpass_filter, downsampling_alpha
from enfify.rodriguez_audio_authenticity import find_cut_in_phases
from enfify.utils import add_defaults, read_wavfile
from enfify.visualization import plot_func

# CONSTANTS
app = typer.Typer()

# FUNCTIONS
@app.command()
def frontend(
    audio_file_path: str = typer.Argument(..., help="The path of the audio file to process."),
    config_path: str = typer.Option(None, help="The path of the configuration file to use."),
):
    """
    ENFify - Audio Tampering Detection Tool

    Args:
        audio_file_path: The path of the audio file to process.
        config_path: The path of the configuration file to use.
    """

    print(f"Processing audio file: {audio_file_path}")

    # Define paths for configuration and defaults
    config_path = config_path or os.path.join(os.path.dirname(__file__), 'config.yml')
    defaults_path = os.path.join(os.path.dirname(__file__), 'defaults.yml')

    # Load configuration
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
    else:
        print(f"Configuration file not found: {config_path}")

    # Load defaults
    defaults = {}
    if os.path.exists(defaults_path):
        try:
            with open(defaults_path, "r") as f:
                defaults = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")

    # Merge defaults into configuration if configuration is empty
    if not config:
        print("Using defaults as no valid configuration file found.")
        config = defaults
    else:
        add_defaults(config, defaults)

    # Read data
    sig, fs = read_wavfile(audio_file_path)

    # Process data with the loaded configuration
    main(sig, fs, config)


def main(sig, fs, config):
    # ENF EXTRACTION

    # Downsampling
    downsample_config = config["downsample"]
    if downsample_config["is_enabled"]:
        f_ds = downsample_config["downsampling_frequency"]

        sig, fs = downsampling_alpha(sig, fs, f_ds)

    # Bandpass Filter
    bandpass_config = config["bandpassfilter"]
    if bandpass_config["is_enabled"]:
        lowcut = bandpass_config["lowcut"]
        highcut = bandpass_config["highcut"]
        sig = bandpass_filter(sig, lowcut, highcut, fs, 1)

    # Variational Mode Decomposition
    vmd_config = config["VMD"]
    if vmd_config["is_enabled"]:
        alpha = vmd_config["alpha"]
        tau = vmd_config["tau"]
        n_mode = vmd_config["n_mode"]
        DC = vmd_config["DC"]
        tol = vmd_config["tol"]

        u_clean, _, _ = VMD(sig, alpha, tau, n_mode, DC, tol)
        sig = u_clean[0]

    # Robust Filtering Algorithm
    rfa_config = config["RFA"]
    if rfa_config["is_enabled"]:
        f0 = rfa_config["f0"]
        I = rfa_config["I"]
        tau = rfa_config["tau"]
        epsilon = rfa_config["epsilon"] 

        sig = RFA(sig, fs, tau, epsilon, I, f0)


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

    hilbert_phases_new, x_hilbert_new, hil_interest_region = find_cut_in_phases(
        hilbert_phases, x_hilbert
    )

    plot_func(x_hilbert, x_hilbert_new, hilbert_phases, hilbert_phases_new, hil_interest_region)


if __name__ == "__main__":
    app()
