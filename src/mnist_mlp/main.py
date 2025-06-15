import numpy as np
import torch
from ..bo import BayesianOptimizer, RBFKernel
from ..searchspace import HyperparamSpace, Real, Integer
from .dataloader import load_mnist
from .mnist_mlp_train import train


def create_search_space() -> HyperparamSpace:
    """Define the hyperparameter search space for the MLP model."""
    #     return HyperparamSpace([
    #     Real("learning_rate", 1e-4, 1e-2, log_prior=True),
    #     Real("dropout_rate", 0.0, 0.5),
    #     Integer("hidden_dim", 32, 256),
    #     Real("weight_decay", 1e-6, 1e-2, log_prior=True),
    #     Integer("n_layers", 2, 4),
    #     Integer("batch_size", 32, 128),
    # ])
    return HyperparamSpace([
        Real("learning_rate", 1e-5, 1e-2, log_prior=True),
        Real("dropout_rate", 0.0, 0.5),
        Integer("hidden_dim", 32, 512),
        Real("weight_decay", 1e-6, 1e-2, log_prior=True),
        #Integer("n_layers", 2, 5),
        Integer("batch_size", 32, 128),
    ])

# well this is implemented but we dont use it yet
user_seed_trials = [
    ({"learning_rate": 1e-3,
      "dropout_rate": 0.2,
      "hidden_dim": 128,
      "weight_decay": 0.0,
      "n_layers": 2,
      "batch_size": 64}, 0.971)   # prev val acc
    ]

def main():
    master_rng = np.random.default_rng(seed=123)
    torch.manual_seed(int(master_rng.integers(0, 2**31 - 1)))


    (X_train, y_train, X_val, y_val, X_test, y_test) = load_mnist(filepath="data/mnist.pkl")
    space = create_search_space()
    opt = BayesianOptimizer(space,
                            kernel=RBFKernel(lengthscale=0.4),
                            maximize=True,
                            visualize=True,
                            viz_slice=("learning_rate", "dropout_rate"),
                            noise=0.002,
                            )

    opt.initialize(
        train, X_train, y_train, X_val, y_val,
        #user_trials=user_seed_trials,
        method="lhs",#"sobol", # or "random"
    )

    best_cfg, best_acc, hist = opt.run(train, X_train, y_train, X_val, y_val, n_iter=15)

    print("✓ Best accuracy", best_acc)
    print("✓ Hyper-parameters", best_cfg)


if __name__ == "__main__":
    main()