import inspect
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import requests
import typer
import yaml
from loguru import logger
from rich import print
from rich.panel import Panel
from rich.text import Text
from scipy.io import wavfile
from typing_extensions import Annotated

from enfify import (
    CONFIG_DIR,
    MODELS_DIR,
    bilstm_classifier,
    cnn_classifier,
    feature_freq_pipeline,
    report,
    sectioning,
)
from enfify.example_files import create_auth_tamp_clip, func_ENF_synthesis_corrupted_harmonic

logger.remove()
logger.add(sys.stderr, level="INFO")
warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)
np.random.seed(0)

app = typer.Typer()

# CONSTANTS
DEFAULT_CONFIG_FILE = CONFIG_DIR / "default.yml"
with open(DEFAULT_CONFIG_FILE, "r") as f:
    DEFAULT = yaml.safe_load(f)
EXAMPLE_OUTDIR = Path(".")


# COMMANDS
# @app.callback(invoke_without_command=True) # TODO: fix or remove
@app.command()
def detect(
    # fmt: off
    audio_file: Annotated[Path, typer.Argument(help="Path to the audio file to detect.")],
    config_file: Annotated[Path, typer.Option(help="Path to a config file.")] = None,
    classifier: Annotated[str, typer.Option(help="Classifier to use for classification (cnn|cnn-bilstm).")] = DEFAULT["classifier"],
    nominal_enf: Annotated[float, typer.Option(help="Nominal ENF frequency.")] = DEFAULT["nominal_enf"],
    downsample_per_enf: Annotated[float, typer.Option(help="Gets multiplied by nominal_enf to obtain the downsample frequency.")] = DEFAULT["downsample_per_enf"],
    bandpass_delta: Annotated[float, typer.Option(help="Bandpass filter delta in Hertz. Added and subtracted from nominal_enf to get the bandpass filter range.")] = DEFAULT["bandpass_delta"],
    frame_len: Annotated[float, typer.Option(help="Frame length for feature calculation in milliseconds.")] = DEFAULT["frame_len"],
    frame_step: Annotated[float, typer.Option(help="Frame step for feature calculation in milliseconds.")] = DEFAULT["frame_step"],
    create_report: Annotated[bool, typer.Option(help="Create a report of the classification.")] = DEFAULT["create_report"],
    # fmt: on
):
    """Program to classify an audiofile as authentic or tampered based on the ENF signal."""

    # Config
    config = select_config(DEFAULT, config_file, locals())

    # Loading
    print(f"[bold white]Analyzing audio file: [bold cyan]{audio_file}[/bold cyan][/bold white]")
    sample_freq, sig = wavfile.read(audio_file)

    if len(sig) / sample_freq < 10:
        logger.warning(
            "The audio file is less than 10 seconds long but model trained with 10 second files. Padding with zeros might affect the result."
        )
        sig = np.pad(sig, (0, 10 * sample_freq - len(sig)))

    # Preprocessing
    feature_freq_vector = feature_freq_pipeline(sig, sample_freq, config)

    # Classification
    if classifier == "cnn":
        logger.debug("Using CNN classifier")
        feature_len = config["feature_len"]
        feature_freq = 1000 / config["frame_step"]
        min_overlap = int(2 * config["frame_len"] / 1000 * feature_freq)
        sections = sectioning(feature_freq_vector, feature_len, min_overlap)

        model_path = MODELS_DIR / "onedcnn_model_carioca_83.pth"
        labels = [cnn_classifier(model_path, section) for section in sections]
        logger.debug(f"Labels: {labels}")
        logger.debug(f"num labels: {len(labels)}")
        prediction = any(labels)
    elif classifier == "cnn-bilstm":
        logger.warning("Using CNN-BiLSTM classifier with unoptimized feature preprocessing.")
        model_path = MODELS_DIR / "cnn_bilstm_alldata_model.pth"
        spatial_scaler_path = MODELS_DIR / "cnn_bilstm_alldata_spatial_scaler.pkl"
        temporal_scaler_path = MODELS_DIR / "cnn_bilstm_alldata_temporal_scaler.pkl"
        prediction = bilstm_classifier(
            feature_freq_vector, config, model_path, spatial_scaler_path, temporal_scaler_path
        )

    # Output
    if prediction:
        print(Panel(Text(text="Tampered", style="bold red"), expand=False))
    else:
        print(Panel(Text(text="Authentic", style="bold green"), expand=False))

    if config["create_report"]:
        report(config, labels)


@app.command()
def example_config():
    """Creates an example config file in the current directory."""
    src = DEFAULT_CONFIG_FILE
    dst = EXAMPLE_OUTDIR / "example_config.yml"
    dst.parent.mkdir(exist_ok=True)

    shutil.copy(src, dst)
    print(f"[bold white]Created example config file: [bold cyan]{dst}[/bold cyan][/bold white]")


@app.command()
def example_synth():
    """Creates example audio files of synthetic data in the current directory.
    One authentic and one tampered."""

    audio_length = 10  # seconds
    max_cut_length = 2  # seconds

    # Pathing
    EXAMPLE_OUTDIR.mkdir(exist_ok=True)

    auth_path = EXAMPLE_OUTDIR / "synthetic-auth.wav"
    tamp_path = EXAMPLE_OUTDIR / "synthetic-tamp.wav"

    # Synthesis
    sample_freq = 44100
    raw_sig, _ = func_ENF_synthesis_corrupted_harmonic(
        fundamental_f=DEFAULT["nominal_enf"],
        duration=audio_length + max_cut_length,
        fs=sample_freq,
    )

    start_ind, cutlen_samples = create_auth_tamp_clip(
        raw_sig, sample_freq, audio_length, max_cut_length, auth_path, tamp_path
    )
    logger.info(f"Saved authentic file: {auth_path.absolute()}")
    logger.info(f"Saved tampered file: {tamp_path.absolute()}")

    cut_info_path = EXAMPLE_OUTDIR / "cut_info.yml"
    cut_info = {
        auth_path.name: {"start": start_ind / sample_freq, "cutlen": cutlen_samples / sample_freq}
    }
    with open(cut_info_path, "w") as f:
        yaml.dump(cut_info, f)
    print(
        f"[bold white]Created synthetic example audio files in folder: [bold cyan]{EXAMPLE_OUTDIR.absolute()}[/bold cyan][/bold white]"
    )


@app.command()
def example_whu():
    """Creates example audio files of WHU_ref data in the current directory.
    One authentic and one tampered."""

    audiolen = 10  # seconds
    max_cutlen = 2  # seconds

    # URL of the permalink
    url = "https://github.com/ghua-ac/ENF-WHU-Dataset/raw/78ed7f3784949f769f291fc1cb94acd10da6322f/ENF-WHU-Dataset/H1_ref/001_ref.wav"

    # Pathing
    EXAMPLE_OUTDIR.mkdir(exist_ok=True)
    auth_path = EXAMPLE_OUTDIR / "whuref-auth.wav"
    temp_dir = Path(tempfile.mkdtemp())
    temp_path = temp_dir / "_.wav"
    tamp_path = EXAMPLE_OUTDIR / "whuref-tamp.wav"

    # Download the file
    logger.info(f"Downloading the file from: {url}")
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Failed to download the file: {response.status_code}")

    with open(temp_path, "wb") as f:
        f.write(response.content)

    logger.info("Saving the file ...")
    sample_freq, sig = wavfile.read(temp_path)

    # Trim the file
    audiolen_samples = audiolen * sample_freq
    max_cutlen_samples = max_cutlen * sample_freq

    # Save the authentic file
    wavfile.write(auth_path, sample_freq, sig[:audiolen_samples])
    logger.info(f"Saved authentic file: {auth_path.absolute()}")

    # Tamper the file
    sig = sig[: audiolen_samples + max_cutlen_samples]
    cutlen_samples = np.random.randint(max_cutlen_samples)
    start_ind = np.random.randint(0, audiolen_samples - 2 * cutlen_samples)
    tamp_sig = np.delete(sig.copy(), slice(start_ind, start_ind + cutlen_samples))
    wavfile.write(tamp_path, sample_freq, tamp_sig)
    logger.info(f"Saved tampered file: {tamp_path.absolute()}")

    cut_info_path = EXAMPLE_OUTDIR / "cut_info.yml"
    cut_info = {
        auth_path.name: {"start": start_ind / sample_freq, "cutlen": cutlen_samples / sample_freq}
    }
    with open(cut_info_path, "w") as f:
        yaml.dump(cut_info, f)
    print(
        f"[bold white]Created example audio files from WHU_ref dataset in folder: [bold cyan]{EXAMPLE_OUTDIR.absolute()}[/bold cyan][/bold white]"
    )


def select_config(default, config_file, _locals):
    # TODO: Refactor to a function
    # prio 3
    config = default.copy()
    # prio 2
    if config_file is not None:
        try:
            with open(config_file, "r") as f:
                update = yaml.safe_load(f)
                config.update(update)
        except Exception as e:
            logger.warning(
                f"Could not load config file: {e}, using CLI Parameter and defaults only."
            )
    # prio 1
    kwargs = {
        k: _locals[k]
        for k in inspect.signature(detect).parameters.keys()
        if _locals[k] != DEFAULT.get(k)
    }
    config.update(kwargs)

    return config


if __name__ == "__main__":
    app()
