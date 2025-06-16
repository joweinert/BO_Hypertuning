import numpy as np
import torch
from ..bo import BayesianOptimizer, RBFKernel, ARDRBFKernel
from .dataloader import load_mnist
from .mnist_mlp_train import train
from ..shared_searchspace import create_search_space

# well this is implemented but we dont use it yet
user_seed_trials = [
    ({"learning_rate": 1e-3,
      "dropout_rate": 0.2,
      "hidden_dim": 128,
      "weight_decay": 0.0,
      "batch_size": 64}, 0.971)   # config_dict, prev val acc -> same structure as hyperparam space -> use previous runs as information for BO
    ]

def main():
    master_rng = np.random.default_rng(seed=0)
    torch.manual_seed(int(master_rng.integers(0, 2**31 - 1)))


    (X_train, y_train, X_val, y_val, X_test, y_test) = load_mnist(filepath="data/mnist.pkl")
    space = create_search_space()
    opt = BayesianOptimizer(space,
                            kernel=ARDRBFKernel,
                            maximize=True,
                            visualize=True,
                            viz_slice=("learning_rate", "dropout_rate"),
                            noise=0.01,
                            xi_schedule=lambda iter: 0.02 * 0.8**iter,
                            rng=master_rng,
                            )

    opt.initialize(
        train, X_train, y_train, X_val, y_val,
        #user_trials=user_seed_trials,
        method="lhs",#"sobol", # or "random"
    )

    best_cfg, best_acc, hist = opt.run(train, X_train, y_train, X_val, y_val, n_iter=15)

    print("Best accuracy", best_acc)
    print("Hyperparameters", best_cfg)


if __name__ == "__main__":
    main()