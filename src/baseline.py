import argparse
import numpy as np
import torch

from .searchspace import HyperparamSpace, Real, Integer
from .mnist_mlp.dataloader import load_mnist
from .mnist_mlp.mnist_mlp_train import train

def create_search_space() -> HyperparamSpace:
    return HyperparamSpace([
        Real("learning_rate", 1e-5, 1e-2, log_prior=True),
        Real("dropout_rate", 0.0, 0.5),
        Integer("hidden_dim", 32, 512),
        Real("weight_decay", 1e-6, 1e-2, log_prior=True),
        #Integer("n_layers", 2, 5),
        Integer("batch_size", 32, 128),
    ])

def baseline(method: str, n_trials: int, seed: int = 123):
    rng = np.random.default_rng(seed)
    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))

    Xtr, ytr, Xv, yv, _, _ = load_mnist(filepath="data/mnist.pkl")
    space = create_search_space()

    cfgs = space.sample_batch(n_trials, rng, method=method)
    best_cfg, best_acc, hist = None, -np.inf, []

    for i, cfg in enumerate(cfgs, 1):
        acc = train(cfg, Xtr, ytr, Xv, yv)
        hist.append(acc)
        if acc > best_acc:
            best_acc, best_cfg = acc, cfg
        print(f"[{method}] Trial {i:02d}/{n_trials}: acc={acc:.4f}  "
              f"best={best_acc:.4f}")

    print("\n=== {0} baseline finished ===".format(method.upper()))
    print(" best accuracy :", best_acc)
    print(" best config   :", best_cfg)
    return best_cfg, best_acc, hist

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random / Sobol / LHS baseline for hyper-parameter search")
    parser.add_argument("--method", default="random",
                        choices=["random", "sobol", "lhs"],
                        help="sampling strategy for the baseline")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="number of configurations to evaluate")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    baseline(args.method, args.n_trials, args.seed)
