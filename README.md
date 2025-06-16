# BO_Hypertuning

In case of testing with MNIST, first the zip in data must be extracted.

Everything should be run from the project root using module notation:

*python -m src.mnist_mlp.main* -> runs bo on MNIST MLP

*python -m src.baseline [--method (one of: random, sobol, lhs)]* -> runs the chosen baseline random approach

(When installing requirements.txt on MAC remove +cpu from pyTorch)