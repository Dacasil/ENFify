import inspect
import sys
import warnings
from pathlib import Path

import numpy as np
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
    butterworth_bandpass_filter,
    cnn_classifier,
    downsample_ffmpeg,
    framing,
    freq_estimation_DFT1,
    report,
    sectioning,
)

warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)
app = typer.Typer()

DEFAULT_FILE = CONFIG_DIR / "default.yml"
with open(DEFAULT_FILE, "r") as f:
    DEFAULT = yaml.safe_load(f)


@app.command()
def main(
    # fmt: off
    audio_file: Annotated[Path, typer.Argument(help="Path to the audio file to detect.")],
    config_file: Annotated[Path, typer.Option(help="Path to a config file.")] = None,
    nominal_enf: Annotated[float, typer.Option(help="Nominal ENF frequency.")] = DEFAULT["nominal_enf"],
    downsample_per_enf: Annotated[float, typer.Option(help="Gets multiplied by nominal_enf to obtain the downsample frequency.")] = DEFAULT["downsample_per_enf"],
    bandpass_delta: Annotated[float, typer.Option(help="Bandpass filter delta in Hertz. Added and subtracted from nominal_enf to get the bandpass filter range.")] = DEFAULT["bandpass_delta"],
    bandpass_order: Annotated[int, typer.Option(help="Bandpass filter order.")] = DEFAULT["bandpass_order"],
    frame_len: Annotated[float, typer.Option(help="Frame length for feature calculation in milliseconds.")] = DEFAULT["frame_len"],
    frame_step: Annotated[float, typer.Option(help="Frame step for feature calculation in milliseconds.")] = DEFAULT["frame_step"],
    window_type: Annotated[str, typer.Option(help="Window type for windowing the frames.")] = DEFAULT["window_type"],
    n_dft: Annotated[int, typer.Option(help="Number of DFT points for frequency estimation.")] = DEFAULT["n_dft"],
    feature_len: Annotated[int, typer.Option(help="Number of features to use for classification.")] = DEFAULT["feature_len"],
    create_report: Annotated[bool, typer.Option(help="Create a report of the classification.")] = DEFAULT["create_report"],
    log_level: Annotated[str, typer.Option(help="Log level for the logger.")] = DEFAULT["log_level"],
    # fmt: on
):  # TODO: -> ...
    """Program to classify an audiofile as authentic or tampered based on the ENF signal."""

    # Config
    # TODO: Refactor to a function
    # TODO: Maybe use a library like dynaconf
    # prio 3
    config = DEFAULT.copy()
    # prio 2
    if config_file is not None:
        try:
            with open(config_file, "r") as f:
                config.update(yaml.safe_load(f))
        except Exception as e:
            logger.warning(
                f"Could not load config file: {e}, using CLI Parameter and defaults only."
            )
    # prio 1
    _locals = locals()
    kwargs = {
        k: _locals[k]
        for k, v in inspect.signature(main).parameters.items()
        if v.default != inspect.Parameter.empty
    }
    config.update(kwargs)

    # Setup
    logger.remove()
    logger.add(sys.stderr, level=config["log_level"])

    print(f"[bold white]Analyzing audio file: [bold cyan]{audio_file}[/bold cyan][/bold white]")
    # Loading
    sample_freq, sig = wavfile.read(audio_file)

    # Downsampling
    downsample_freq = config["downsample_per_enf"] * config["nominal_enf"]
    sig, sample_freq = downsample_ffmpeg(sig, sample_freq, downsample_freq)

    # Bandpass Filter
    lowcut = config["nominal_enf"] - config["bandpass_delta"]
    highcut = config["nominal_enf"] + config["bandpass_delta"]
    bandpass_order = config["bandpass_order"]
    sig = butterworth_bandpass_filter(sig, sample_freq, lowcut, highcut, bandpass_order)

    # Frame Splitting
    window_type = config["window_type"]
    frame_len = config["frame_len"]
    frame_shift = config["frame_step"]
    frames = framing(sig, sample_freq, frame_len, frame_shift, window_type)

    # Frequency Estimation
    n_dft = config["n_dft"]
    window_type = config["window_type"]
    feature_freqs = np.array(
        [freq_estimation_DFT1(frame, sample_freq, n_dft, window_type) for frame in frames]
    )

    # Classification
    feature_len = config["feature_len"]
    feature_freq = 1000 / config["frame_step"]
    min_overlap = int(2 * config["frame_len"] / 1000 * feature_freq)
    sections = sectioning(feature_freqs, feature_len, min_overlap)

    labels = [cnn_classifier(feature_freqs) for section in sections]

    # Output
    if any(labels):
        result = "Tampered"
        style = "bold red"
    else:
        result = "Authentic"
        style = "bold green"
    # fmt: off
    print(Panel(Text(text=result, style=style, justify="center"), expand=False))
    # fmt: on

    if config["create_report"]:
        report(config, labels)


if __name__ == "__main__":
    app()
