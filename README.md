# ENFify

Welcome to **ENFify**, an open-source tool to detect digital audio tampering using the Electric Network Frequency background noise.

## Table of Contents
- [Description](#description)
- [Status](#status)
- [Installation](#installation)
- [Testing](#testing)
- [Organization](#organization)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Description
The goal of the **ENFify** project is to develop an open-source tool for analyzing audio recordings using the Electric Network Frequency (ENF) signal contained in the audio. This tool aims to enable users to precisely distinguish between authentic (uncut) and tampered (cut) audio files. By analysing the phase of the ENF signal, the tool will help detect audio manipulations based on the background noise of the ENF.

## Status
**ENFify** is currently a work in progress and is not yet available as a stable release. The project is under active development, and features and functionality may change as we work towards the first stable version scheduled for release on September 30th, 2024. We have released an alpha version of the tool that can be tested with specific datasets featuring minimal noise. Exemplary audio files are included in the repository.

## Installation
Step-by-step instructions on how to get the development environment running.

1. **Clone the repository**
```bash
git clone https://github.com/Licates/ENFify.git
```

2. **Navigate to the project directory**
```bash
cd ENFify
```

3. **Create and activate the Conda environment**
```bash
make requirements
make data
conda activate enfify
```

## Testing

To test the alpha version, enter the following command.

```bash
python3 enfify/enfify_alpha.py data/samples/whu_cut_min_001_ref.wav
```

There are also other sample files in the [`samples`](samples) folder. The `whu*.wav` files there are sourced from the [ENF-WHU-Dataset](https://github.com/ghua-ac/ENF-WHU-Dataset/tree/78ed7f3784949f769f291fc1cb94acd10da6322f/ENF-WHU-Dataset/H1_ref). These files include reference power grid data, therefore with minimal noise. They were truncated to one minute and in the `*cut*` files added with a deletion cut.

The `synthetic*.wav` files contain synthetic data. For details refer to [`Synthetic_Data.ipynb`](notebooks/ls/Synthetic_Data.ipynb).

The output files of the test are stored in the folder [`reports`](reports).

## Organization
```
├── LICENSE            <- MIT license.
│
├── README.md          <- The top-level README.
│
├── samples            <- Sample audio files for testing enfify.
│
├── notebooks          <- Jupyter notebooks grouped by developer.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in 
|                         reporting.
│
├── tests              <- Folder for unittests. Not yet implemented.
│
├── unused_ccds        <- Unused file and folder templates from CCDS.
│
├── environment.yml    <- The environment file for reproducing the analysis
|                         environment with conda.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         enfify and configuration for tools like black.
│
├── setup.cfg          <- Configuration file for flake8.
│
└── enfify             <- Source code for use in this project.
    │
    ├── __init__.py          <- Makes enfify a Python module
    │
    ├── enfify_alpha.py      <- Alpha version of the ENFify tool.
    │
    ├── config.yml           <- Config file to manually overwrite the default
    |                           config.
    │
    ├── defaults.yml         <- Default config file.
    │
    ├── visualization.py     <- Module with functions to create results
    |                           oriented visualizations.
    │
    └── ...                  <- Further modules mainly organized as the steps
                                of the process.
```

## Contributing
Thank you for your interest in contributing to this project! Contributions will be welcomed starting from October 1st, 2024. This project is being developed as part of a Summer of Code program, and the first stable version is scheduled for release on September 30th, 2024. Until then, the codebase is under active development, and major changes are expected. We appreciate your patience and look forward to your contributions once the stable release is available.

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute the code as long as you include the original copyright and license notice in any copies or substantial portions of the software. For more details, please refer to the [LICENSE](LICENSE) file.

## References
Several academic papers and resources have been considered in the development of this project. Here are some of the key references:

1. **1D-CNN-based audio tampering detection using ENF signals** \
*Author(s): Haifeng Zhao, Yanming Ye, Xingfa Shen, Lili Liu \
Journal: Scientific Reports \
DOI: https://doi.org/10.1038/s41598-024-60813-0*

2. **Audio Authenticity: Detecting ENF Discontinuity With High Precision Phase Analysis** \
*Author(s): D. P. Nicolalde Rodriguez, J. A. Apolinario and L. W. P. Biscainho \
Journal: IEEE Transactions on Information Forensics and Security \
DOI: https://doi.org/10.1109/TIFS.2010.2051270*

We would like to thank the authors of these papers for their valuable contributions, which served as a foundation for our implementation.

--------