import numpy as np
import os
import yaml
import typer
from enfify.enf_enhancement import VMD, RFA
from enfify.enf_estimation import segmented_phase_estimation_DFT0, segmented_phase_estimation_hilbert
from enfify.preprocessing import bandpass_filter, downsampling_alpha
from enfify.rodriguez_audio_authenticity import find_cut_in_phases
from enfify.utils import add_defaults, read_wavfile
from enfify.visualization import create_cut_phase_plot, create_phase_plot, cut_to_alpha_pdf, to_alpha_pdf

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

    # DFT0 instantaneous phase estimation
    phases = segmented_phase_estimation_DFT0(sig, fs, num_cycles, n_dft, nom_enf)
    x_DFT0 = np.linspace(0.0, time, len(phases))

    hilbert_phases_new, x_hilbert_new, hil_interest_region = find_cut_in_phases(
        hilbert_phases, x_hilbert
    )
    
    DFT0_phases_new, x_DFT0_new, DFT0_interest_region = find_cut_in_phases(phases, x_DFT0)

    # Create the phase plots
    # TODO: Paths in config or as terminal arguments
    root_path = os.path.join(os.path.dirname(__file__), "../reports")
    hilbert_phase_path = f"{root_path}/figures/hilbert_phase_im.png"
    hilbert_cut_phase_path = f"{root_path}/figures/cut_hilbert_phase_im.png"
    DFT0_phase_path = f"{root_path}/figures/DFT0_phase_im.png"
    DFT0_cut_phase_path = f"{root_path}/figures/cut_DFT0_phase_im.png"
    pdf_outpath = f"{root_path}/enfify_alpha.pdf"

    if not np.any(hil_interest_region) and not np.any(DFT0_interest_region):
        create_phase_plot(x_hilbert, hilbert_phases, hilbert_phase_path)
        create_phase_plot(x_DFT0, phases, DFT0_phase_path)
        to_alpha_pdf(hilbert_phase_path, hilbert_phase_path, pdf_outpath)

    elif not np.any(hil_interest_region) and np.any(DFT0_interest_region):
        create_phase_plot(x_hilbert, hilbert_phases, hilbert_phase_path)
        create_phase_plot(x_DFT0, phases, DFT0_phase_path)
        create_cut_phase_plot(
            x_DFT0_new, DFT0_phases_new, x_DFT0, DFT0_interest_region[0], DFT0_cut_phase_path
        )
        to_alpha_pdf(hilbert_phase_path, DFT0_cut_phase_path, pdf_outpath)

    elif np.any(hil_interest_region) and not np.any(DFT0_interest_region):
        create_phase_plot(x_hilbert, hilbert_phases, hilbert_phase_path)
        create_cut_phase_plot(
            x_hilbert_new,
            hilbert_phases_new,
            x_hilbert,
            hil_interest_region[0],
            hilbert_cut_phase_path,
        )
        create_phase_plot(x_DFT0, phases, DFT0_phase_path)
        to_alpha_pdf(hilbert_cut_phase_path, DFT0_phase_path, pdf_outpath)

    else:
        create_phase_plot(x_hilbert, hilbert_phases, hilbert_phase_path)
        create_cut_phase_plot(
            x_hilbert_new,
            hilbert_phases_new,
            x_hilbert,
            hil_interest_region[0],
            hilbert_cut_phase_path,
        )

        create_phase_plot(x_DFT0, phases, DFT0_phase_path)
        create_cut_phase_plot(
            x_DFT0_new,
            DFT0_phases_new,
            x_DFT0,
            DFT0_interest_region[0],
            DFT0_cut_phase_path,
        )

        cut_to_alpha_pdf(hilbert_cut_phase_path, DFT0_cut_phase_path, pdf_outpath)

if __name__ == "__main__":
    app()
