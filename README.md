# nn-flash: Neural Networks for Flash Calculations



## Main script help

```
python main.py --help

usage: NN Flash [-h] [-s {3,30,300}] [-r] [-cv] [-t] [-a] [-viz] [-p] [-w] [-v]

Neural Networks for Flash Calculations.
https://github.com/antonionicampos/nn-flash

options:
  -h, --help            show this help message and exit
  -s {3,30,300}, --samples-per-composition {3,30,300}
                        Select dataset depending on number of P, T samples per composition sample
  -r, --read            Read, transform and process raw data
  -cv, --cross-validation
                        Create CV datasets from processed data
  -t, --training        Do train step
  -a, --analysis        Do Analysis Step
  -viz, --visualization
                        Create and save visualizations
  -p, --phase-diagram   Create Phase Diagram for a sample composition and classification model probability
  -w, --warning         Run in WARNING mode
  -v, --verbose         Run in DEBUG mode
```