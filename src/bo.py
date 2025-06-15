from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional, Tuple


import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

import matplotlib.pyplot as plt
from matplotlib import animation, colors
from sklearn.decomposition import PCA
import seaborn as sns, pandas as pd, numpy as np

from .searchspace import HyperparamSpace


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
        return [(-5, 5), (-5, 5)]

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
        """Exact GP fit by computing K + σ²I and its Cholesky factor."""
        X = np.atleast_2d(X).astype(np.float64)
        y = np.atleast_1d(y).astype(np.float64)
        assert X.shape[0] == y.shape[0]
        K = self.kernel(X) + (self.noise**2) * np.eye(len(X))
        self._L = np.linalg.cholesky(K + 1e-12 * np.eye(len(K)))
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y))
        self.X_train, self.y_train = X, y
        return self

    def predict(self, X_star: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, np.ndarray | None]:
        """Predict mean (and optionally stddev) at test points."""
        X_star = np.atleast_2d(X_star).astype(np.float64)
        assert self._L is not None, "Call fit() first."
        K_star = self.kernel(self.X_train, X_star)
        mean = K_star.T @ self._alpha  # (N*,)
        if not return_std:
            return mean, None
        v = np.linalg.solve(self._L, K_star)
        K_ss = self.kernel(X_star, None)
        var = np.diag(K_ss) - np.sum(v**2, axis=0)
        var = np.maximum(var, 1e-12)  # numerical safety
        std = np.sqrt(var)
        return mean, std

    def nll(self, params: np.ndarray) -> float:
        """Return NLML for given log‑lengthscale and log‑variance."""
        length, var = np.exp(params)
        self.kernel.lengthscale = length
        self.kernel.variance = var
        K = self.kernel(self.X_train) + (self.noise**2) * np.eye(len(self.X_train))
        L = np.linalg.cholesky(K + 1e-12 * np.eye(len(K)))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
        nll = 0.5 * self.y_train.T @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * len(K) * np.log(2 * np.pi)
        return float(nll)


# Acquisition: Expected Improvement

def expected_improvement(
    X: np.ndarray,
    gp: GaussianProcess,
    f_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Analytic EI for *minimisation* problems."""
    mu, sigma = gp.predict(X, return_std=True)
    sigma = sigma + 1e-12
    z = (f_best - mu - xi) / sigma

    ei = (f_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    return np.maximum(ei, 0.0)


# Acquisition optimiser (search over unit hyper‑cube)

def propose_location(
    gp: GaussianProcess,
    bounds: list[tuple[float, float]],
    y_best: float,
    rng: Optional[np.random.Generator] = None,
    n_restarts: int = 25,
    n_raw_samples: int = 1000,
    xi: float = 0.01,
) -> np.ndarray:
    """
    Suggest the next evaluation point by maximising Expected Improvement.

    Returns
    -------
    np.ndarray, shape (D,)
        Normalised coordinates of the candidate in [0, 1]^D.
    """
    dim = len(bounds)
    rng = np.random.default_rng() if rng is None else rng

    # EI as a *negative* function so we can use scipy’s minimiser
    def _neg_ei(z: np.ndarray) -> float:
        return -expected_improvement(z[None, :], gp, y_best, xi=xi)[0]

    # ---- stage 1: random search to find good starting points ---------------
    Z_raw = rng.uniform(0.0, 1.0, size=(n_raw_samples, dim))
    starts = Z_raw[np.argsort([_neg_ei(z) for z in Z_raw])[:n_restarts]]

    # ---- stage 2: local L-BFGS-B polish -----------------------------------
    best_x, best_val = None, np.inf
    for x0 in starts:
        res = minimize(_neg_ei, x0=x0, bounds=bounds, method="L-BFGS-B")
        if res.fun < best_val:
            best_x, best_val = res.x, res.fun

    assert best_x is not None
    return best_x


class BayesianOptimizer:
    """
    Simple GP-based Bayesian optimiser.

    Parameters
    ----------
    space : HyperparamSpace
        Search space describing the hyper-parameters.
    kernel : KernelBase | type[KernelBase]
        Either a kernel instance or a class (in which case *kernel_kwargs*
        are forwarded to the constructor).
    maximize : bool
        True  -> maximise the returned metric (e.g. accuracy)
        False -> minimise it         (e.g. loss)
    noise : float
        Observation noise σ.
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
        noise: float = 1e-6,
        relearn_kernel: bool = True,
        n_relearn = 3,
        visualize: bool = False, 
        viz_slice: tuple[str,str] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.space = space
        self.rng = np.random.default_rng() if rng is None else rng
        self.maximize = maximize
        self.relearn_kernel = relearn_kernel
        self.n_relearn = n_relearn
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
        """Add user-supplied trials then top-up to D with new designs."""
        self.X, self.y = np.empty((0, len(self.space))), np.empty((0,))

        # 1) ingest user-supplied evaluations (skip duplicates)
        seen: set[tuple] = set()
        if user_trials:
            for cfg, metric in user_trials:
                z = self.space.to_vector(cfg)
                key = tuple(z.round(12))        # hashable
                if key in seen:                 # duplicate
                    continue
                seen.add(key)
                loss = -metric if self.maximize else metric
                self._append(cfg, loss)

        # 2) decide how many extra points we need
        target = max(len(self.space), 1)
        n_extra = target - len(self.y)
        if n_extra > 0:
            extra_cfgs = self.space.sample_batch(n_extra, self.rng, method=method)
            for cfg in extra_cfgs:
                metric = train_fn(cfg, X_train, y_train, X_val, y_val)
                loss = -metric if self.maximize else metric
                self._append(cfg, loss)

        best_metric = max(-self.y) if self.maximize else self.y.min()
        print(f"Initialised with {len(self.y)} points (user + {n_extra} new). "
              f"Best metric so far: {best_metric:.4f}")


    def _relearn_kernel(self):
        if self.gp.X_train is None:
            return
        if len(self.y) % self.n_relearn: # only relearn every n_relearn iterations
            return
        def nll(vec): return self.gp.nll(vec)
        x0 = self.kernel.hyperparams()
        res = minimize(nll, x0, bounds=self.kernel.bounds())
        self.kernel.set_hyperparams(res.x)   # gp.nll updates kernel, but explicit is fine

    def ask(self, iter_cnt, xi_schedule=lambda iter: 0.01*0.8**iter) -> np.ndarray:
        """Return the next normalised vector proposed by Expected Improvement."""
        assert len(self.y) > 0, "Call initialize() first."
        self.gp.fit(self.X, self.y)
        if self.relearn_kernel:
            self._relearn_kernel()
        
        xi = xi_schedule(iter_cnt)

        z_next = propose_location(
            self.gp,
            bounds=self.space.bounds(),
            y_best=float(self.y.min()),
            rng=self.rng,
            n_restarts=25,
            n_raw_samples=1000,
            xi=xi,
        )
        return z_next

    def tell(
        self,
        z: np.ndarray,
        metric: float,
    ) -> None:
        """Feed a finished evaluation back into the optimiser."""
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
        """Full BO loop: repeatedly ask → train_fn → tell."""
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
            self.visualize_bo() 
        return best_cfg, best_metric, history

    # ------------------------------------------------------------------ internals
    def _append(self, cfg: Dict[str, Any], loss: float) -> None:
        self.X = np.vstack([self.X, self.space.to_vector(cfg)[None, :]])
        self.y = np.append(self.y, loss)
        self.snapshots.append((self.X.copy(), self.y.copy()))


    def visualize_bo(self, out_dir=".", dpi=120):
        if not self.visualize:
            return

        # ---------- 3.1 PCA scatter ---------------------------------------------
        Z = self.X
        metric = -self.y if self.maximize else self.y
        p = PCA(n_components=2).fit_transform(Z)
        plt.figure(figsize=(6,5))
        mu, sigma = metric.mean(), metric.std()
        norm = colors.Normalize(vmin=mu - sigma, vmax=mu + sigma)
        sc = plt.scatter(p[:,0], p[:,1], c=metric, cmap="viridis", norm=norm, s=40, alpha=0.8)
        plt.colorbar(sc, label="metric")
        plt.title("PCA of sampled hyper-parameters")
        plt.annotate("best", xy=p[np.argmax(metric)])
        plt.savefig(f"{out_dir}/pca_scatter.png", dpi=dpi)
        plt.close()

        # ---------- 3.2 seaborn pairplot ----------------------------------------
        df = pd.DataFrame({
            **{name: self.space.denormalise_col(name, Z[:, k])
               for k, name in enumerate(self.space.param_names)},
            "metric": metric,
        })
        q = 5  # number of quantiles for binning
        df["metric_bin"] = pd.qcut(df["metric"], q=q, labels=False)

        sns.pairplot(
            df,
            vars=self.space.param_names,
            hue="metric_bin",
            palette="viridis",
            diag_kind="kde"
        )
        plt.savefig(f"{out_dir}/pairplot.png", dpi=dpi)
        plt.close()

        # ---------- 3.3 GIF of 2-D slice ----------------------------------------
        dim1, dim2 = self.viz_slice
        i, j = map(self.space.index_of, (dim1, dim2))
        grid = np.linspace(0.0, 1.0, 100)
        G1, G2 = np.meshgrid(grid, grid)
        Zgrid = np.zeros((grid.size*grid.size, Z.shape[1]))

        fig, ax = plt.subplots()
        img = ax.imshow(np.zeros_like(G1), extent=[0,1,0,1], origin="lower",
                cmap="plasma", vmin=0, vmax=1)          # dummy clim
        scat = ax.scatter([], [], c="white", s=15)

        def update(frame):
            Xf, yf = self.snapshots[frame]
            self.gp.fit(Xf, yf)
            Zgrid[:] = 0.5
            Zgrid[:, i] = G1.ravel()
            Zgrid[:, j] = G2.ravel()
            mu, std = self.gp.predict(Zgrid, return_std=True)   # << change
            img.set_data(std.reshape(G1.shape))
            img.set_clim(vmin=0, vmax=std.max())
            scat.set_offsets(Xf[:, [i, j]])        # update points
            ax.set_title(f"iteration {frame}")
            return img, scat

        ani = animation.FuncAnimation(fig, update,
                                      frames=len(self.snapshots),
                                      interval=600, blit=False)
        ani.save(f"{out_dir}/surrogate.gif", writer="pillow")
        plt.close(fig)
        print(f"Visualisations saved to {out_dir}")