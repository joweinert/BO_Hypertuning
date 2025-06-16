from .searchspace import HyperparamSpace, Real, Integer

def create_search_space() -> HyperparamSpace:
    """Defines the hyperparameter search space for the MLP model AND baselines."""
    return HyperparamSpace([
        Real("learning_rate", 1e-5, 1e-2, log_prior=True),
        Real("dropout_rate", 0.0, 0.5),
        Integer("hidden_dim", 32, 512),
        Real("weight_decay", 1e-6, 1e-2, log_prior=True),
        Integer("batch_size", 32, 128),
    ])