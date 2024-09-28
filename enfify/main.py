import inspect
import sys
import warnings
from pathlib import Path

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
    cnn_classifier,
    bilstm_classifier,
    feature_freq_pipeline,
    report,
    sectioning,
    plot_feature_freq,
)

logger.remove()
logger.add(sys.stderr, level="INFO")
warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

app = typer.Typer()

DEFAULT_CONFIG_FILE = CONFIG_DIR / "default.yml"
with open(DEFAULT_CONFIG_FILE, "r") as f:
    DEFAULT = yaml.safe_load(f)


# @app.callback(invoke_without_command=True)
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

    # Preprocessing
    feature_freq_vector = feature_freq_pipeline(sig, sample_freq, config)

    # Classification
    if classifier == "cnn":
        feature_len = config["feature_len"]
        feature_freq = 1000 / config["frame_step"]
        min_overlap = int(2 * config["frame_len"] / 1000 * feature_freq)
        sections = sectioning(feature_freq_vector, feature_len, min_overlap)

        model_path = MODELS_DIR / "onedcnn_model_carioca_83.pth"
        labels = [cnn_classifier(model_path, section) for section in sections]
        prediction = any(labels)
    elif classifier == "cnn-bilstm":
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
def configfile():
    raise NotImplementedError("Not implemented yet.")


@app.command()
def synthexample():
    raise NotImplementedError("Not implemented yet.")


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
