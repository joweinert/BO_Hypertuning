from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional, Tuple


import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

import numpy as np

from .searchspace import HyperparamSpace
from .viz import visualize


# Kernels

class KernelBase(ABC):
    @abstractmethod
    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray: ...

    @abstractmethod
    def hyperparams(self) -> np.ndarray: ...

    @abstractmethod
    def set_hyperparams(self, vec: np.ndarray) -> None: ...

    @abstractmethod
    def bounds(self) -> list[tuple[float, float]]: ...
        
@dataclass
class RBFKernel(KernelBase):
    lengthscale: float = 1.0
    variance: float = 1.0

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        """Compute the squared-exponential kernel matrix.

        Args:
            X: (..., D)
            Y: (N, D) or None (defaults to X)
        Returns:
            (M, N) kernel matrix
        """
        if Y is None:
            Y = X
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        sqdist = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
        return self.variance * np.exp(-0.5 * sqdist / self.lengthscale**2)

    def hyperparams(self): # log-space
        return np.log([self.lengthscale, self.variance])
    
    def set_hyperparams(self, vec):
        self.lengthscale, self.variance = np.exp(vec)

    def bounds(self):
        # log-space bounds: lengthscale in [e-2, e^0.5], variance in [e-2, e^5]
        return [(-2, 0.5), (-2, 5)]
    


class ARDRBFKernel:
    """
    Separates length-scale per dimension
    bounds() returns one (lengthscale_d) pair per dim plus one for log variance.
    """
    def __init__(self, lengthscale: np.ndarray | float | None = None, variance: float = 1.0):
        """
        If lengthscale is None we start with 1.0 for every dim, the optimizers
        heuristic in initialize() overwrites it once D is known.
        """
        self.lengthscale = np.asarray(lengthscale if lengthscale is not None else 1.0, dtype=float)
        self.variance = float(variance)

    # kernel matrix
    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        if Y is None:
            Y = X
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        diff = (X[:, None, :] - Y[None, :, :]) / self.lengthscale
        sqdist = np.sum(diff ** 2, axis=-1)
        return self.variance * np.exp(-0.5 * sqdist)

    def hyperparams(self) -> np.ndarray: # log-space
         return np.log(np.concatenate([self.lengthscale, [self.variance]]))

    def set_hyperparams(self, vec: np.ndarray) -> None:
         self.lengthscale = np.exp(vec[:-1])
         self.variance    = float(np.exp(vec[-1]))

    # bounds for NLML optimiser: each log10 lengthscale_d in [−2,0.5], log10 variance in [−2,5]
    def bounds(self):
        D = self.lengthscale.size
        return [(-2, 0.5)] * D + [(-2, 5)]

# Gaussian Process regression

class GaussianProcess:
    def __init__(self, kernel: Callable[[np.ndarray, np.ndarray | None], np.ndarray], noise: float = 1e-6):
        self.kernel = kernel
        self.noise = noise
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self._L: np.ndarray | None = None  # Cholesky factor
        self._alpha: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        """Exact GP fit by computing K + variance*I and its Cholesky factor."""
        X = np.atleast_2d(X).astype(np.float64)
        y = np.atleast_1d(y).astype(np.float64)
        assert X.shape[0] == y.shape[0]
        self.y_mean = y.mean()
        y_cent      = y - self.y_mean
        K = self.kernel(X) + (self.noise**2) * np.eye(len(X))
        self._L = np.linalg.cholesky(K + 1e-12 * np.eye(len(K)))
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y_cent))
        self.X_train, self.y_train = X, y
        return self

    def predict(self, X_star: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, np.ndarray | None]:
        """Predict mean (and optionally stddev) at test points."""
        X_star = np.atleast_2d(X_star).astype(np.float64)
        assert self._L is not None, "Call fit() first."
        K_star = self.kernel(self.X_train, X_star)
        mean = K_star.T @ self._alpha + self.y_mean
        if not return_std:
            return mean, None
        v = np.linalg.solve(self._L, K_star)
        K_ss = self.kernel(X_star, None)
        var = np.diag(K_ss) - np.sum(v**2, axis=0)
        var = np.maximum(var, 1e-12)  # numerical safety
        std = np.sqrt(var)
        return mean, std

    def nll(self, log_params: np.ndarray) -> float:
        """
        Negative log marginal likelihood for any kernel that implements
        set_hyperparams().  We pass log-space params straight through.
        """
        self.kernel.set_hyperparams(log_params)
        K = self.kernel(self.X_train) + (self.noise**2) * np.eye(len(self.X_train))
        L = np.linalg.cholesky(K + 1e-12 * np.eye(len(K)))
        y_cent = self.y_train - self.y_mean
        alpha  = np.linalg.solve(L.T, np.linalg.solve(L, y_cent))
        nll = 0.5 * y_cent.T @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * len(K) * np.log(2 * np.pi)
        return float(nll)


# Acquisition: Expected Improvement

def expected_improvement(
    X: np.ndarray,
    gp: GaussianProcess,
    f_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """EI at points X, given the GP model gp and the best observed value f_best."""
    mu, sigma = gp.predict(X, return_std=True)
    sigma = np.maximum(sigma, 2e-2) 
    z = (f_best - mu - xi) / sigma

    ei = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(ei, 0.0)


def propose_location(
    gp: GaussianProcess,
    bounds: list[tuple[float, float]],
    y_best: float,
    rng: Optional[np.random.Generator] = None,
    n_restarts: int = 25,
    n_raw_samples: int = 1000,
    xi: float = 0.01,
) -> np.ndarray:
    dim = len(bounds)
    rng = np.random.default_rng() if rng is None else rng

    # computes EI on a big random grid
    Z_raw = rng.uniform(0.0, 1.0, size=(n_raw_samples, dim))
    ei_vals = expected_improvement(Z_raw, gp, y_best, xi=xi)
    print(f"[DEBUG] EI over {n_raw_samples} random points:")
    print(f"        min={ei_vals.min():.3e},  max={ei_vals.max():.3e}")
    print(f"        mean={ei_vals.mean():.3e},  #>0 = {int((ei_vals>0).sum())}/{n_raw_samples}")

    mu, sigma = gp.predict(Z_raw, return_std=True)
    mask = sigma < 1e-4
    ei_vals[mask] = 0.0
    sigma[mask]   = 1e-4 
    print(f"[DEBUG] σ over random points: min={sigma.min():.3e}, max={sigma.max():.3e}, mean={sigma.mean():.3e}")
    ls = gp.kernel.lengthscale
    if np.ndim(ls) == 0 or np.isscalar(ls):
        ls_str = f"{float(ls):.3f}"
    else:
        ls_str = np.array2string(ls, precision=3)
    print(f"[DEBUG] kernel ls={ls_str}, var={gp.kernel.variance:.3f}")

    # negative acquisition for L-BFGS
    def _neg_ei(z: np.ndarray) -> float:
        return -expected_improvement(z[None, :], gp, y_best, xi=xi)[0]

    # best few starting points by EI
    starts = Z_raw[np.argsort([_neg_ei(z) for z in Z_raw])[:n_restarts]]

    # polish with LBFGSB
    best_x, best_val = None, np.inf
    for x0 in starts:
        res = minimize(_neg_ei, x0=x0, bounds=bounds, method="L-BFGS-B")
        if res.fun < best_val:
            best_x, best_val = res.x, res.fun

    assert best_x is not None
    print(f"[DEBUG] best EI = {-best_val:.3e} at z = {best_x}")
    return best_x

class BayesianOptimizer:
    """
    Simple GP-based Bayesian optimizer.

    Parameters
    ----------
    space : HyperparamSpace
        Search space describing the hyper-parameters.
    kernel : KernelBase | type[KernelBase]
        Either a kernel instance or a class (in which case *kernel_kwargs*
        are forwarded to the constructor).
    maximize : bool
        True  -> maximise the returned metric (e.g. accuracy)
        False -> minimise it (e.g. loss)
    noise : float
        Observation noise sigma (std not variance!).
    rng : np.random.Generator | None
        Random generator used for initial designs and restarts.
    """
    def __init__(
        self,
        space: HyperparamSpace,
        *,
        kernel: KernelBase | type[KernelBase] = RBFKernel,
        kernel_kwargs: dict | None = None,
        maximize: bool = False,
        noise: float = 0.003,
        relearn_kernel: bool = False,
        n_relearn = 2,
        xi_schedule: Callable[[int], float] = lambda iter: 0.02,
        visualize: bool = False, 
        viz_slice: tuple[str,str] = None,
        rng: Optional[np.random.Generator] = None,
        outdir: str = "assets",
    ):
        self.space = space
        self.rng = np.random.default_rng() if rng is None else rng
        self.outdir = outdir
        self.maximize = maximize
        self.relearn_kernel = relearn_kernel
        self.n_relearn = n_relearn
        self.xi_schedule = xi_schedule
        self.visualize = visualize
        assert (not visualize) or (viz_slice is not None), "viz_slice must be set when visualize is True"
        self.viz_slice = viz_slice
        self.snapshots: list[tuple[np.ndarray, np.ndarray]] = []

        self.kernel = (
            kernel(**(kernel_kwargs or {})) if isinstance(kernel, type) else kernel
        )
        self.gp = GaussianProcess(self.kernel, noise=noise)

        # running data set
        self.X: np.ndarray = np.empty((0, len(space)))
        self.y: np.ndarray = np.empty((0,))

    def initialize(
        self,
        train_fn,
        X_train, y_train, X_val, y_val,
        *,
        user_trials: list[tuple[dict, float]] | None = None,
        method: str = "sobol",
    ):
        """Adding user-supplied trials then top-up to 3xD with new designs."""
        self.X, self.y = np.empty((0, len(self.space))), np.empty((0,))

        #ingests user-supplied evaluations
        seen: set[tuple] = set()
        if user_trials:
            for cfg, metric in user_trials:
                z = self.space.to_vector(cfg)
                key = tuple(z.round(12))
                if key in seen: # duplicate
                    continue
                seen.add(key)
                loss = -metric if self.maximize else metric
                self._append(cfg, loss)
        # sample new configurations -> up to 3D
        target = max(3*len(self.space), 1)
        n_extra = target - len(self.y)
        if n_extra > 0:
            extra_cfgs = self.space.sample_batch(n_extra, self.rng, method=method)
            for cfg in extra_cfgs:
                metric = train_fn(cfg, X_train, y_train, X_val, y_val)
                loss = -metric if self.maximize else metric
                self._append(cfg, loss)

        # lengthscale_d to median pair-wise distance in each dimension,
        # variance to empirical variance of y. Works for both RBF and ARD-RBF.
        if len(self.y) >= 2:
            Z = self.X
            # median L1 distance per dimension, ensure 1D array
            med_ls = np.median(np.abs(Z[:, None, :] - Z[None, :, :]), axis=(0, 1))
            med_ls = np.clip(med_ls, 0.05, 5.0)
            med_ls = np.atleast_1d(med_ls) # scalar for isotropic RBF

            # empirical variance of losses, lower-bounded
            var_y = max(float(np.var(self.y)), 0.01)

            # packingg log-hyper-params
            if isinstance(self.kernel, RBFKernel):
                ls0 = float(np.median(np.abs(Z[:, None, :] - Z[None, :, :])))
                hp_vec = np.log([ls0, var_y])
            elif isinstance(self.kernel, ARDRBFKernel):
                hp_vec = np.log(np.concatenate([med_ls, [var_y]]))
                self.kernel.set_hyperparams(hp_vec)
            else:
                raise TypeError(f"Unsupported kernel type: {type(self.kernel)}")

        best_metric = max(-self.y) if self.maximize else self.y.min()
        print(f"Initialized with {len(self.y)} points (user + {n_extra} new). "
              f"Best metric so far: {best_metric:.4f}")


    def _relearn_kernel(self):
        if self.gp.X_train is None:
            return
        if len(self.y) < 4 * len(self.space):  # need gt 4D points for stability
            return
        if len(self.y) % self.n_relearn:
            return
        def nll(vec): return self.gp.nll(vec)
        x0 = self.kernel.hyperparams()
        res = minimize(nll, x0, bounds=self.kernel.bounds())
        if res.fun > nll(x0): # skip if optimization diverged
            return
        self.kernel.set_hyperparams(res.x) # gp.nll updates kernel, but explicit is fine


    def _propose(self, iter_cnt):
        self.gp.fit(self.X, self.y)
        if self.relearn_kernel:
            self._relearn_kernel()
        xi = self.xi_schedule(iter_cnt)
        return propose_location(
            self.gp, self.space.bounds(), float(self.y.min()),
            rng=self.rng, n_restarts=25, n_raw_samples=1000, xi=xi)

    def ask(self, iter_cnt):
        for _ in range(7): # up to 7 attempts
            z = self._propose(iter_cnt)
            if not (self.X.size
                    and (np.abs(self.X - z).max(axis=1) < 1e-3).any()):
                return z # unique
        # fallback: random jitter
        return self.rng.uniform(0.0, 1.0, size=len(self.space))

    def tell(
        self,
        z: np.ndarray,
        metric: float,
    ) -> None:
        """Feeding a finished evaluation back into the optimizer."""
        loss = -metric if self.maximize else metric
        self._append(self.space.from_vector(z), loss)

    def run(
        self,
        train_fn: Callable[[Dict[str, Any],
                            np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        *,
        n_iter: int = 30,
    ) -> tuple[Dict[str, Any], float, List[float]]:
        """Full BO loop: repeatedly ask -> train_fn -> tell."""
        history: List[float] = []
        for t in range(1, n_iter + 1):
            z = self.ask(t)
            cfg = self.space.from_vector(z)
            metric = train_fn(cfg, X_train, y_train, X_val, y_val)
            self.tell(z, metric)
            history.append(metric)

            best_loss = self.y.min()
            best_so_far = -best_loss if self.maximize else best_loss
            print(f"Iter {t:02d}: metric={metric:.4f}, best={best_so_far:.4f}")

        best_idx = int(self.y.argmin())
        best_cfg = self.space.from_vector(self.X[best_idx])
        best_metric = -self.y[best_idx] if self.maximize else self.y[best_idx]
        if self.visualize:
            visualize(
                X        = self.X,
                y        = self.y,
                space    = self.space,
                maximize = self.maximize,
                snapshots=self.snapshots,
                viz_slice=self.viz_slice,
                out_dir  = self.outdir,
            )
        return best_cfg, best_metric, history

    def _append(self, cfg: Dict[str, Any], loss: float, eps: float = 1e-3) -> None:
        """
        Stores the new sample unless its vector is already inside
        an epsilon-cube of an existing point, where esilon = 1e-3 means points that differ
        by < 0.1 % in every dimension are considered duplicates -> EI is
        still free to probe the neighbourhood.
        """
        z = self.space.to_vector(cfg)
        if self.X.size:
            dmax = np.abs(self.X - z).max(axis=1)
            if (dmax < eps).any():
                return # exact duplicate
        self.X = np.vstack([self.X, z[None, :]])
        self.y = np.append(self.y, loss)
        self.snapshots.append((self.X.copy(), self.y.copy()))