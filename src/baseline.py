# import argparse
# import numpy as np
# import torch

# from .searchspace import HyperparamSpace, Real, Integer
# from .mnist_mlp.dataloader import load_mnist
# from .mnist_mlp.mnist_mlp_train import train
# from .viz import visualize

# def create_search_space() -> HyperparamSpace:
#     return HyperparamSpace([
#         Real("learning_rate", 1e-5, 1e-2, log_prior=True),
#         Real("dropout_rate", 0.0, 0.5),
#         Integer("hidden_dim", 32, 512),
#         Real("weight_decay", 1e-6, 1e-2, log_prior=True),
#         #Integer("n_layers", 2, 5),
#         Integer("batch_size", 32, 128),
#     ])

# def baseline(method: str, n_trials: int, seed: int = 123):
#     rng = np.random.default_rng(seed)
#     torch.manual_seed(int(rng.integers(0, 2**31 - 1)))

#     Xtr, ytr, Xv, yv, _, _ = load_mnist(filepath="data/mnist.pkl")
#     space = create_search_space()

#     cfgs = space.sample_batch(n_trials, rng, method=method)
#     best_cfg, best_acc, hist = None, -np.inf, []

#     for i, cfg in enumerate(cfgs, 1):
#         acc = train(cfg, Xtr, ytr, Xv, yv)
#         hist.append(acc)
#         if acc > best_acc:
#             best_acc, best_cfg = acc, cfg
#         print(f"[{method}] Trial {i:02d}/{n_trials}: acc={acc:.4f}  "
#               f"best={best_acc:.4f}")

#     print("\n=== {0} baseline finished ===".format(method.upper()))
#     print(" best accuracy :", best_acc)
#     print(" best config   :", best_cfg)
#     visualize(
#         X        = np.vstack(vecs),              # all sampled z-vectors
#         y        = np.array([-acc for acc in scores]),  # same sign convention
#         space    = space,
#         maximize = True,                         # or False if you stored loss
#         snapshots=None,                          # no GP → no GIF
#         viz_slice=None
#     )
#     return best_cfg, best_acc, hist

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Random / Sobol / LHS baseline for hyper-parameter search")
#     parser.add_argument("--method", default="random",
#                         choices=["random", "sobol", "lhs"],
#                         help="sampling strategy for the baseline")
#     parser.add_argument("--n_trials", type=int, default=20,
#                         help="number of configurations to evaluate")
#     parser.add_argument("--seed", type=int, default=123)
#     args = parser.parse_args()

#     baseline(args.method, args.n_trials, args.seed)
    
# baseline_main.py  ------------------------------------------------------
"""
Baseline driver for uniform-random, Sobol, or Latin-Hypercube sampling.
Runs `n_trials` configurations and saves PCA + pair-plot figures.
"""

import argparse
import numpy as np
import torch

from .mnist_mlp.dataloader import load_mnist
from .mnist_mlp.mnist_mlp_train import train
from .searchspace import HyperparamSpace, Real, Integer
from .viz import visualize
from .shared_searchspace import create_search_space


def baseline(method: str, n_trials: int, seed: int = 123):
    rng = np.random.default_rng(seed)
    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))

    Xtr, ytr, Xv, yv, _, _ = load_mnist(filepath="data/mnist.pkl")
    space = create_search_space()

    # randomly sample `n_trials` configurations from the search space according to method 
    cfgs = space.sample_batch(n_trials, rng, method=method)
    vecs, scores = [], []

    best_cfg, best_acc = None, -np.inf
    for i, cfg in enumerate(cfgs, 1):
        acc = train(cfg, Xtr, ytr, Xv, yv)
        vecs.append(space.to_vector(cfg))
        scores.append(acc)

        if acc > best_acc:
            best_acc, best_cfg = acc, cfg

        print(f"[{method}] Trial {i:02d}/{n_trials}: acc={acc:.4f}  "
              f"best={best_acc:.4f}")

    print(f"\n=== {method.upper()} baseline finished ===")
    print(" best accuracy :", best_acc)
    print(" best config   :", best_cfg)

    visualize(
        X        = np.vstack(vecs),
        y        = -np.array(scores), # flipped -> maximize accuracy
        space    = space,
        maximize = True,
        snapshots=None, # skip GIF
        viz_slice=None,
        out_dir  = f"fig_{method}"
    )
    return best_cfg, best_acc, scores


# ------------------------------------------------------------------- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Uniform / Sobol / LHS baseline for hyper-parameter search")
    parser.add_argument("--method", default="random",
                        choices=["random", "sobol", "lhs"],
                        help="sampling strategy")
    parser.add_argument("--n_trials", type=int, default=20,
                        help="number of configurations to evaluate")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    baseline(args.method, args.n_trials, args.seed)
