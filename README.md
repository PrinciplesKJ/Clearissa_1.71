# Clearissa

## Rollout of the Clearissa Data Analysis Software

Clearissa is a data analysis platform for multi-channel fluorescence spectroscopy. It streamlines data import from plate readers, provides visualisation tools, and supports kinetic modelling workflows for DNA and RNA strand displacement fluorescence and FRET experiments.

## Features

* Import raw plate reader exports (CSV or Excel) for multi-channel fluorescence experiments
* Interactive visualisation and data frame inspection tools
* Kinetic analysis workflows for fluorescence and FRET experiments
* Export-ready tables and plots

## Requirements

* Python 3.9-3.11
* Qt5 runtime via PyQt5

Full Python dependencies are listed in `requirements.txt` and `pyproject.toml`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

## Running Clearissa

```bash
python core/Clearissa_main.py
```

Tip: If running from a packaged build, use the provided executable instead of the Python entry point.

## Building an Executable

Clearissa includes convenience scripts for packaging with PyInstaller:

* Windows: `build_clearissa.bat`
* macOS/Linux: `build_clearissa.sh`
* Cross-platform helper: `build_executable.py`

Example:

```bash
./build_clearissa.sh
```

## Documentation

* `manual.html` contains the user manual
* Configuration files are located in the `config/` directory

## Contact

For more information, feedback, or suggestions, contact:
[k.jurinovic22@imperial.ac.uk](mailto:k.jurinovic22@imperial.ac.uk)
