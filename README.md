# nn-flash: Neural Networks for Flash Calculations

This work aims to investigate the calculation of vapor-liquid equilibrium flash for oil reservoirs from the perspective of computational intelligence. In other words, to use data-based models to assist or replace conventional flash calculations.

## Pre-requisites

- [Python 3.10.11](https://www.python.org/downloads/release/python-31011/)
- [Java JDK 21 LTS](https://adoptium.net/temurin/releases/)
- [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#latest-microsoft-visual-c-redistributable-version) for Windows users

## Setup

From the project folder:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Main script help

```bash
$ python main.py --help
usage: NN Flash [-h] [--task {classification,regression,synthesis}] [--regression-loss {mse,mse_with_constraint}] [-r] [-cv] [-t] [-a] [-v] [-g]

Neural Networks for Flash Calculations.
https://github.com/antonionicampos/nn-flash

options:
  -h, --help            show this help message and exit
  --task {classification,regression,synthesis}
                        Task(s) to run pipeline
  --regression-loss {mse,mse_with_constraint}
                        Regression loss function
  -r, --read            Read, transform and process raw data
  -cv, --cross-validation
                        Create CV datasets
  -t, --training        Do train step
  -a, --analysis        Do Analysis Step
  -v, --viz             Create and save visualizations
  -g, --generate        Synthesize new samples
```
