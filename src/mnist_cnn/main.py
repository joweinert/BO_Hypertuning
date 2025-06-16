import numpy as np
import torch
from ..bo import BayesianOptimizer, RBFKernel
from ..searchspace import HyperparamSpace, Real, Integer
from ..mnist_mlp.dataloader import load_mnist
from .mnist_cnn_train import train

# CNN search space
def create_search_space() -> HyperparamSpace:
    """Define the hyperparameter search space for the CNN model."""
    return HyperparamSpace([
        Real("learning_rate", 1e-5, 1e-2, log_prior=True),
        Real("dropout_rate", 0.0, 0.5),
        Integer("conv_channels_1", 16, 128),
        Integer("conv_channels_2", 16, 128),
        Real("weight_decay", 1e-6, 1e-2, log_prior=True),
        Integer("kernel_size", 2, 5),
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
                            rng=master_rng,
                            )

    opt.initialize(
        train, X_train, y_train, X_val, y_val,
        #user_trials=user_seed_trials,
        method="lhs",#"sobol", # or "random"
    )

    best_cfg, best_acc, hist = opt.run(train, X_train, y_train, X_val, y_val, n_iter=15)

    print("✓ Best accuracy", best_acc)
    print("✓ Hyper-parameters", best_cfg)


# Main CNN function
def main_cnn():
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
                            rng=master_rng,
                            )

    def train_wrapper(hyper_cfg, X_train, y_train, X_val, y_val, device="cpu"):
        cnn_cfg = {
            "learning_rate": hyper_cfg["learning_rate"],
            "dropout_rate": hyper_cfg["dropout_rate"],
            "conv_channels": [int(hyper_cfg["conv_channels_1"]), int(hyper_cfg["conv_channels_2"])],
            "weight_decay": hyper_cfg["weight_decay"],
            "kernel_size": int(hyper_cfg["kernel_size"]),
            "batch_size": int(hyper_cfg["batch_size"]),
        }
        return train(cnn_cfg, X_train, y_train, X_val, y_val, device=device)

    opt.initialize(
        train_wrapper, X_train, y_train, X_val, y_val,
        method="lhs",
    )

    best_cfg, best_acc, hist = opt.run(train_wrapper, X_train, y_train, X_val, y_val, n_iter=15)

    print("✓ Best accuracy (CNN)", best_acc)
    print("✓ Hyper-parameters (CNN)", best_cfg)


if __name__ == "__main__":
    main()