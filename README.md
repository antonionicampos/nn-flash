# nn-flash: Neural Networks for Flash Calculations

This work aims to investigate the calculation of vapor-liquid equilibrium flash for oil reservoirs from the perspective of computational intelligence. In other words, to use data-based models to assist or replace conventional flash calculations.

## Pre-requisites

- Python 3.10.11

## Setup

From the project folder:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Main script help

```
python main.py --help

usage: NN Flash [-h] [-s {3,30}] --task {classification,regression,synthesis} [--regression-loss {mse,mse_with_constraint}] [-r] [-cv] [-t] [-a] [-v]

Neural Networks for Flash Calculations.
https://github.com/antonionicampos/nn-flash

options:
  -h, --help            show this help message and exit
  -s {3,30}, --samples-per-composition {3,30}
                        Select dataset depending on number of P, T samples per composition sample
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
```
