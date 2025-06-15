# # viz.py -----------------------------------------------------------------
# from typing import List, Tuple, Optional
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib import colors, animation
# import seaborn as sns
# from sklearn.decomposition import PCA
# import datetime, os

# def visualize(
#     X: np.ndarray,                      # (N, D) in [0,1]
#     y: np.ndarray,                      # loss if minimise, −metric if maximise
#     space,                              # HyperparamSpace instance
#     maximize: bool,
#     *,
#     snapshots: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,  # BO only
#     viz_slice: Optional[Tuple[str, str]] = None,      # ("lr", "dropout")
#     out_dir: str = ".",
#     dpi: int = 120,
# ) -> None:
#     """
#     Produce PCA scatter, pair-plot (with log-axes where needed),
#     and—if `snapshots` is provided—an animated GIF of GP uncertainty
#     on the 2-D slice `viz_slice`.

#     Parameters
#     ----------
#     X, y      :  All evaluated points (normalised) and their stored losses
#     space     :  The search-space object (needs .param_names, .denormalise_col,
#                  .index_of)
#     maximize  :  If True, colours are flipped back to metric = −y
#     snapshots :  List of (X_subset, y_subset) after every evaluation; pass None
#                  for baselines so the GIF is skipped.
#     viz_slice :  Two parameter names to define the GIF axes
#     """
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir) 
#     ts = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
#     # ------------------------------------------------------------------ colours
#     metric = -y if maximize else y
#     mu, sigma = metric.mean(), metric.std()
#     norm = colors.Normalize(vmin=mu - sigma, vmax=mu + sigma)

#     # ------------------------------------------------------------------ PCA ---
#     p = PCA(n_components=2).fit_transform(X)
#     plt.figure(figsize=(6, 5))
#     sc = plt.scatter(p[:, 0], p[:, 1], c=metric,
#                      cmap="viridis", norm=norm, s=40, alpha=0.8)
#     plt.colorbar(sc, label="metric")
#     plt.annotate("best", xy=p[np.argmax(metric)])
#     plt.title("PCA of sampled hyper-parameters")
#     plt.savefig(f"{out_dir}/{ts}_pca_scatter.png", dpi=dpi)
#     plt.close()

#     # ----------------------------------------------------------- pair-plot ----
#     raw_df = pd.DataFrame({
#         **{name: space.denormalise_col(name, X[:, k])
#            for k, name in enumerate(space.param_names)},
#         "metric": metric,
#     })

#     # put log axes for learning_rate and weight_decay
#     cols_log = {"learning_rate", "weight_decay"} & set(space.param_names)
#     plot_df = raw_df.copy()
#     for col in cols_log:
#         plot_df[col] = np.log10(plot_df[col])
#         plot_df.rename(columns={col: f"log10({col})"}, inplace=True)

#     # bin metric into quantiles for colour
#     plot_df["qbin"] = pd.qcut(plot_df["metric"], q=5, labels=False)
#     g = sns.pairplot(
#         plot_df,
#         vars=[c for c in plot_df.columns if c not in {"metric", "qbin"}],
#         hue="qbin",
#         palette="viridis",
#         diag_kind="kde"
#     )
#     g.savefig(f"{out_dir}/{ts}_pairplot.png", dpi=dpi)
#     plt.close()

#     # ------------------------------------------------------------- GIF ------
#     if snapshots and viz_slice:
#         dim1, dim2 = viz_slice
#         i, j = map(space.index_of, (dim1, dim2))
#         grid = np.linspace(0.0, 1.0, 100)
#         G1, G2 = np.meshgrid(grid, grid)
#         Zgrid = np.zeros((grid.size * grid.size, X.shape[1]))

#         fig, ax = plt.subplots()
#         img = ax.imshow(np.zeros_like(G1), extent=[0, 1, 0, 1],
#                         origin="lower", cmap="plasma", vmin=0, vmax=1)
#         scat = ax.scatter([], [], c="white", s=15)

#         def update(frame):
#             Xf, yf = snapshots[frame]
#             # GP is part of the BO object that created snapshots; fit on the fly
#             from bo import GaussianProcess, RBFKernel          # lazy import
#             gp = GaussianProcess(kernel=RBFKernel(), noise=1e-6).fit(Xf, yf)

#             Zgrid[:] = 0.5
#             Zgrid[:, i] = G1.ravel()
#             Zgrid[:, j] = G2.ravel()
#             _, std = gp.predict(Zgrid, return_std=True)
#             img.set_data(std.reshape(G1.shape))
#             img.set_clim(vmin=0, vmax=std.max())
#             scat.set_offsets(Xf[:, [i, j]])
#             ax.set_title(f"iteration {frame}")
#             return img, scat

#         ani = animation.FuncAnimation(fig, update, frames=len(snapshots),
#                                       interval=600, blit=False)
#         ani.save(f"{out_dir}/{ts}_surrogate.gif", writer="pillow")
#         plt.close(fig)

#     print(f"[visualize] figures saved to: {out_dir}")


# viz.py -----------------------------------------------------------------
from typing import List, Tuple, Optional
import os, datetime, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors, animation
from sklearn.decomposition import PCA
import seaborn as sns


def visualize(
    X: np.ndarray,                      # (N, D) in [0,1]
    y: np.ndarray,                      # loss if minimise, −metric if maximise
    space,                              # HyperparamSpace (needs .param_names, .denormalise_col, .index_of)
    maximize: bool,
    *,
    snapshots: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    viz_slice: Optional[Tuple[str, str]] = None,
    out_dir: str = ".",
    dpi: int = 120,
) -> None:
    """
    Save PCA scatter, pair-plot, and—if snapshots are given—an animated
    GIF of GP uncertainty on a 2-D slice.

    Pass `snapshots=None` to skip the GIF (baseline runs).
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%d%m%Y-%H%M%S")

    metric = -y if maximize else y
    mu, sigma = metric.mean(), metric.std()
    colour_norm = colors.Normalize(vmin=mu - sigma, vmax=mu + sigma)

    #PCA scatter
    p = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(p[:, 0], p[:, 1], c=metric,
                     cmap="viridis", norm=colour_norm, s=40, alpha=0.8)
    plt.colorbar(sc, label="metric")
    plt.annotate("best", xy=p[np.argmax(metric)])
    plt.title("PCA of sampled hyper-parameters")
    plt.savefig(f"{out_dir}/{ts}_pca_scatter.png", dpi=dpi)
    plt.close()

    # pair plot
    df_raw = pd.DataFrame({
        **{name: space.denormalise_col(name, X[:, k])
           for k, name in enumerate(space.param_names)},
        "metric": metric,
    })

    log_cols = {"learning_rate", "weight_decay"} & set(space.param_names)
    df_plot = df_raw.copy()
    for col in log_cols:
        df_plot[col] = np.log10(df_plot[col])
        df_plot.rename(columns={col: f"log10({col})"}, inplace=True)

    df_plot["qbin"] = pd.qcut(df_plot["metric"], q=5, labels=False)
    g = sns.pairplot(
        df_plot,
        vars=[c for c in df_plot.columns if c not in {"metric", "qbin"}],
        hue="qbin",
        palette="viridis",
        diag_kind="kde"
    )
    g.savefig(f"{out_dir}/{ts}_pairplot.png", dpi=dpi)
    plt.close()

    # optional gif (doesnt work for random baseline but will work if viz_slice is set in BO)
    if snapshots and viz_slice and len(snapshots) > 1:
        dim1, dim2 = viz_slice
        i, j = map(space.index_of, (dim1, dim2))
        grid = np.linspace(0.0, 1.0, 100)
        G1, G2 = np.meshgrid(grid, grid)
        Zgrid = np.zeros((grid.size * grid.size, X.shape[1]))

        fig, ax = plt.subplots()
        img = ax.imshow(np.zeros_like(G1), extent=[0, 1, 0, 1],
                        origin="lower", cmap="plasma", vmin=0, vmax=1)
        scat = ax.scatter([], [], c="white", s=15)

        def update(frame):
            Xf, yf = snapshots[frame]

            # lightweight GP fit on the fly
            from .bo import GaussianProcess, RBFKernel
            gp = GaussianProcess(RBFKernel(), noise=1e-6).fit(Xf, yf)

            Zgrid[:] = 0.5
            Zgrid[:, i] = G1.ravel()
            Zgrid[:, j] = G2.ravel()
            _, std = gp.predict(Zgrid, return_std=True)

            img.set_data(std.reshape(G1.shape))
            img.set_clim(vmin=0, vmax=std.max())
            scat.set_offsets(Xf[:, [i, j]])
            ax.set_title(f"iteration {frame}")
            return img, scat

        ani = animation.FuncAnimation(fig, update,
                                      frames=len(snapshots),
                                      interval=600, blit=False)
        ani.save(f"{out_dir}/{ts}_surrogate.gif", writer="pillow")
        plt.close(fig)

    print(f"[visualize] figures saved to {out_dir}")
