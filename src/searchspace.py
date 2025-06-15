from abc import ABC, abstractmethod
import numpy as np
from typing import Sequence, List, Dict, Any
from scipy.stats import qmc 


class HyperParam(ABC):
    """Abstract base class for a single hyperparam."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def transform(self, value: Any) -> float:
        """Map raw value -> normalised value in [0,1]."""

    @abstractmethod
    def inverse(self, z: float) -> Any:
        """Inverse of transform: normalised -> raw."""

    @abstractmethod
    def sample(self, rng: np.random.Generator | np.random.RandomState | None = None) -> Any:
        """Draw a random sample according to the parameter prior."""

    def get_type(self) -> str:
        return self.__class__.__name__.lower()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"



class Real(HyperParam):

    def __init__(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log_prior: bool = False,
        round_to_int: bool = False,
    ) -> None:
        super().__init__(name)
        assert high > low, "high must be greater than low."
        self.low = float(low)
        self.high = float(high)
        self.log_prior = bool(log_prior)
        self.round_to_int = bool(round_to_int)

    def _log_bounds(self) -> tuple[float, float]:
        """Return log10-scaled bounds."""
        return np.log10(self.low), np.log10(self.high)

    def transform(self, value: float) -> float:
        """Affine warp (uniform) or affine warp of log10(value) (log-uniform)."""
        if self.log_prior:
            log_val = np.log10(value)
            low, high = self._log_bounds()
            return (log_val - low) / (high - low)
        return (value - self.low) / (self.high - self.low)

    def inverse(self, z: float) -> float | int:
        assert 0.0 <= z <= 1.0, "z must be in [0,1]"
        if self.log_prior:
            low, high = self._log_bounds()
            log_val = z * (high - low) + low
            value: float | int = 10 ** log_val
        else:
            value = z * (self.high - self.low) + self.low
        if self.round_to_int:
            value = int(round(value))
        return value

    def sample(
        self,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> float | int:
        rng = np.random.default_rng() if rng is None else rng
        if self.log_prior:
            low, high = self._log_bounds()
            value = 10 ** rng.uniform(low, high)
        else:
            value = rng.uniform(self.low, self.high)
        if self.round_to_int:
            value = int(round(value))
        return value

    def __repr__(self) -> str:  # pragma: no cover
        prior = "log-uniform" if self.log_prior else "uniform"
        rflag = ", int" if self.round_to_int else ""
        return f"Real({self.name}, {self.low}, {self.high}, {prior}{rflag})"


class Integer(Real):
    """Integer variable encoded as real in the GP and rounded when decoded."""

    def __init__(self, name: str, low: int, high: int):
        super().__init__(name, float(low), float(high), log_prior=False, round_to_int=True)


class HyperparamSpace:
    """A *fixed* collection of HyperParam objects."""

    def __init__(self, params: Sequence[HyperParam]):
        self.params: List[HyperParam] = list(params)
        self.param_names: List[str] = self.names()

    def __repr__(self) -> str:  # pragma: no cover
        return "HyperparamSpace(" + ", ".join(repr(p) for p in self.params) + ")"

    def __len__(self) -> int:
        return len(self.params)

    def names(self) -> List[str]:
        return [p.name for p in self.params]
    
    def index_of(self, name: str) -> int:
        return self.param_names.index(name)

    def to_vector(self, config: Dict[str, Any]) -> np.ndarray:
        """Convert raw config dict-> normalised np.ndarray of shape (D,)."""
        return np.array([p.transform(config[p.name]) for p in self.params], dtype=np.float64)

    def from_vector(self, z: Sequence[float]) -> Dict[str, Any]:
        z = np.asarray(z, dtype=np.float64)
        assert z.shape == (len(self),), "Vector dimension mismatch."
        return {p.name: p.inverse(val) for p, val in zip(self.params, z)}
    
    def sample_batch(
        self,
        n: int,
        rng: np.random.Generator,
        method: str = "sobol", # "sobol" | "lhs" | "random"
    ) -> list[dict]:
        D = len(self)
        if method == "sobol":
            engine = qmc.Sobol(d=D, scramble=True, seed=rng.integers(2**32))
            Z = engine.random(n)
        elif method == "lhs":
            engine = qmc.LatinHypercube(d=D, seed=rng.integers(2**32))
            Z = engine.random(n)
        else:                               # fallback random
            Z = rng.random((n, D))
        return [self.from_vector(z) for z in Z]


    def bounds(self) -> List[tuple[float, float]]:
        """Return box bounds in the normalised space - always (0,1) at this stage."""
        return [(0.0, 1.0)] * len(self.params)
    
    def denormalise_col(self, name: str, col: np.ndarray) -> np.ndarray:
        """Vectorised inverse transform for a single column."""
        p = next(par for par in self.params if par.name == name)
        return np.vectorize(p.inverse, otypes=[float])(col)