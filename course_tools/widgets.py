"""Interfaces interactives (widgets) des séances de régression.

Usage dans un notebook :
    from course_tools import widgets
    widgets.manual_line_fit(X_train["carat"], y_train)

- manual_line_fit : deux curseurs pour ajuster une droite à la main (NB02) ;
- gradient_descent : la descente de gradient en 3D, avec un curseur pour le taux d'apprentissage η (NB02) ;
- polynomial_degree : le surapprentissage quand le degré monte (NB03) ;
- ridge_slider : la régularisation Ridge qui écrase les coefficients (NB03) ;
- knn_boundary : la frontière des k plus proches voisins selon k (NB04) ;
- tree_depth : un arbre de décision de plus en plus profond (NB04) ;
- kmeans_step_by_step : k-means étape par étape, avec un départ au hasard (NB05).

Elles servent à introduire une notion devant la classe, puis chacun peut les manipuler.
"""

from __future__ import annotations

import ipywidgets as w
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

TEAL, ANIS, RED, GREY = "#0F8C8C", "#B3C011", "#A8453C", "#7C7A7B"


def _errors(y, y_pred):
    mse = float(np.mean((y - y_pred) ** 2))
    mae = float(np.mean(np.abs(y - y_pred)))
    return mse, mae


def manual_line_fit(x, y, n_points: int = 400, seed: int = 0):
    """Deux curseurs (pente a, ordonnée b) pour ajuster prix = a × carat + b à la main.

    L'erreur quadratique moyenne (MSE) et l'erreur absolue moyenne (MAE) s'affichent en direct.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) > n_points:
        idx = np.random.default_rng(seed).choice(len(x), n_points, replace=False)
        x, y = x[idx], y[idx]
    a_opt, b_opt = np.polyfit(x, y, 1)

    slope = w.FloatSlider(value=2000, min=0, max=15000, step=100, description="pente a", continuous_update=True,
                          style={"description_width": "70px"}, layout=w.Layout(width="420px"))
    intercept = w.FloatSlider(value=0, min=-8000, max=4000, step=100, description="ordonnée b", continuous_update=True,
                             style={"description_width": "70px"}, layout=w.Layout(width="420px"))
    solution = w.Checkbox(value=False, description="montrer la droite trouvée par la machine")

    def draw(a, b, show_fit):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.scatter(x, y, s=10, alpha=0.4, color=TEAL)
        xs = np.linspace(0, x.max() * 1.05, 50)
        ax.plot(xs, a * xs + b, color=ANIS, lw=3, label=f"votre droite : prix = {a:.0f} × carat {b:+.0f}")
        if show_fit:
            ax.plot(xs, a_opt * xs + b_opt, color=RED, lw=2, ls="--",
                    label=f"la machine : prix = {a_opt:.0f} × carat {b_opt:+.0f}")
        mse, mae = _errors(y, a * x + b)
        ax.set_title(f"MSE = {mse:,.0f}   ·   MAE = {mae:,.0f} $".replace(",", " "), fontsize=13)
        ax.set_xlabel("carat")
        ax.set_ylabel("prix ($)")
        ax.set_ylim(-2000, max(y) * 1.05)
        ax.legend(loc="upper left")
        plt.show()

    output = w.interactive_output(draw, {"a": slope, "b": intercept, "show_fit": solution})
    display(w.VBox([slope, intercept, solution]), output)


def _run_descent(x, y, eta, n_max):
    """La descente de gradient du NB02 (départ a = b = 0), arrêtée si elle diverge.

    Renvoie le chemin des (a, b) et la MSE à chaque itération.
    """
    a = b = 0.0
    path, costs = [(a, b)], []
    for _ in range(n_max):
        error = a * x + b - y
        costs.append(float(np.mean(error ** 2)))
        if not np.isfinite(costs[-1]) or costs[-1] > 1e15:
            break
        a, b = a - eta * 2 * np.mean(error * x), b - eta * 2 * np.mean(error)
        path.append((a, b))
    return np.array(path[: len(costs)]), np.array(costs)


def gradient_descent(x, y, n_points: int = 400, seed: int = 0, n_max: int = 20000):
    """La descente de gradient sur prix = a × carat + b, avec un curseur pour le taux d'apprentissage η.

    Trois vues : le chemin sur la surface J(a, b) en 3D, la MSE à chaque itération, et la droite obtenue.
    Le titre donne le nombre d'itérations nécessaires pour arriver au minimum (à 1 % près).
    La descente utilise tous les points ; le nuage n'en affiche que n_points.
    """
    from matplotlib import cm

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    shown = np.arange(len(x))
    if len(x) > n_points:
        shown = np.random.default_rng(seed).choice(len(x), n_points, replace=False)

    # la surface J(a, b), en millions de $², autour du minimum (forme développée : rapide sur une grille)
    sx, sy, sxx, sxy, syy = x.mean(), y.mean(), (x ** 2).mean(), (x * y).mean(), (y ** 2).mean()

    def cost(a, b):
        return (a ** 2 * sxx + b ** 2 + syy + 2 * a * b * sx - 2 * a * sxy - 2 * b * sy) / 1e6

    a_min, b_min = np.polyfit(x, y, 1)
    j_min = cost(a_min, b_min) * 1e6
    half_width = max(8500.0, 1.15 * abs(a_min), 1.15 * abs(b_min))
    A, B = np.meshgrid(np.linspace(a_min - half_width, a_min + half_width, 60),
                       np.linspace(b_min - half_width, b_min + half_width, 60))
    J = cost(A, B)

    cache = {}

    def descent(eta):
        if eta not in cache:
            cache[eta] = _run_descent(x, y, eta, n_max)
        return cache[eta]

    eta = w.SelectionSlider(options=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 0.55, 0.6, 0.7], value=0.1,
                            description="η", continuous_update=False,
                            style={"description_width": "70px"}, layout=w.Layout(width="420px"))
    steps = w.SelectionSlider(options=[0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
                               value=20, description="itérations", continuous_update=False,
                               style={"description_width": "70px"}, layout=w.Layout(width="420px"))

    def draw(lr, n):
        path, costs = descent(lr)
        diverges = len(costs) < n_max and costs[-1] > costs[0]
        reached = np.flatnonzero(costs <= 1.01 * j_min)
        k = min(n, len(costs) - 1)   # dernière itération affichée
        a, b = path[k]

        def fmt_int(v):
            return f"{v:,}".replace(",", " ")

        if diverges:
            summary, color = f"η = {lr} : ça diverge, la MSE augmente à chaque itération", RED
        elif len(reached):
            summary, color = f"η = {lr} : minimum atteint (à 1 % près) en {fmt_int(reached[0])} itérations", TEAL
        else:
            summary, color = f"η = {lr} : toujours pas au minimum après {fmt_int(n_max)} itérations", RED

        fig = plt.figure(figsize=(15, 4.8))
        grid = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1])
        fig.suptitle(summary, color=color, fontsize=15, fontweight="bold")

        # 1. le chemin sur la surface J(a, b) (les points sortis de la surface ne sont pas tracés)
        ax1 = fig.add_subplot(grid[0], projection="3d", computed_zorder=False)
        ax1.plot_surface(A, B, J, cmap=cm.GnBu_r, alpha=0.75, linewidth=0)
        pa, pb = path[: k + 1, 0].copy(), path[: k + 1, 1].copy()
        inside = (np.abs(pa - a_min) <= half_width) & (np.abs(pb - b_min) <= half_width)
        inside &= cost(pa, pb) <= J.max()
        pa[~inside], pb[~inside] = np.nan, np.nan
        ax1.plot(pa, pb, cost(pa, pb), color=RED, lw=2)
        ax1.scatter(pa, pb, cost(pa, pb), color=RED, s=10, depthshade=False)
        ax1.text(0, 0, cost(0, 0) + 15, "départ ", color=RED, fontsize=10, ha="right")
        ax1.scatter([a_min], [b_min], [cost(a_min, b_min)], color="black", marker="*", s=150, depthshade=False)
        ax1.set_xlabel("a")
        ax1.set_ylabel("b")
        ax1.set_zlabel("J(a, b) (M$²)")
        ax1.set_zlim(0, J.max())
        ax1.view_init(elev=22, azim=-60)
        ax1.tick_params(labelsize=7)
        ax1.set_title(f"le chemin sur J(a, b) : {fmt_int(k)} itérations", fontsize=11)

        # 2. la MSE à chaque itération
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(np.arange(k + 1), costs[: k + 1], color=TEAL, lw=2)
        ax2.axhline(j_min, color="black", ls=":", lw=1, label="minimum de J")
        if len(reached) and reached[0] <= k:
            ax2.axvline(reached[0], color=ANIS, lw=2, label=f"minimum atteint ({fmt_int(reached[0])} it.)")
        ax2.set_yscale("log")
        ax2.set_xlim(0, max(k, 1))
        ax2.set_xlabel("itération")
        ax2.set_ylabel("MSE (échelle log)")
        ax2.legend(loc="upper right", fontsize=9)
        ax2.set_title("MSE = " + fmt_int(round(costs[k])), fontsize=11)

        # 3. la droite obtenue
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.scatter(x[shown], y[shown], s=10, alpha=0.4, color=TEAL)
        if np.isfinite(a) and np.isfinite(b) and abs(a) < 1e7:
            xs = np.linspace(0, x.max() * 1.05, 50)
            ax3.plot(xs, a * xs + b, color=ANIS, lw=3)
            ax3.set_title(f"prix = {a:.0f} × carat {b:+.0f}", fontsize=11)
        ax3.set_ylim(-2000, max(y) * 1.05)
        ax3.set_xlabel("carat")
        ax3.set_ylabel("prix ($)")
        fig.tight_layout()
        plt.show()

    output = w.interactive_output(draw, {"lr": eta, "n": steps})
    display(w.VBox([eta, steps]), output)


def polynomial_degree(x, y, n_train: int = 30, seed: int = 0, max_degree: int = 12):
    """Un curseur de degré : la courbe colle aux points d'entraînement, l'erreur de test remonte.

    Les points de test sont pris dans la même plage de carats que l'entraînement : l'erreur qui
    explose ne vient donc pas d'une extrapolation, mais des oscillations de la courbe entre les points.
    """
    import warnings

    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    tr, te = idx[:n_train], idx[n_train:]
    te = te[(x[te] >= x[tr].min()) & (x[te] <= x[tr].max())][:1000]
    degrees = list(range(1, max_degree + 1))
    err_tr, err_te, models = [], [], {}
    for d in degrees:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = make_pipeline(MinMaxScaler((-1, 1)), PolynomialFeatures(d), LinearRegression()).fit(x[tr, None], y[tr])
        models[d] = m
        err_tr.append(np.mean(np.abs(m.predict(x[tr, None]) - y[tr])))
        err_te.append(np.mean(np.abs(m.predict(x[te, None]) - y[te])))

    slider = w.IntSlider(value=1, min=1, max=max_degree, description="degré",
                          style={"description_width": "70px"}, layout=w.Layout(width="420px"))

    def draw(d):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.scatter(x[te], y[te], s=6, alpha=0.15, color=GREY, label="test (jamais vus)")
        ax1.scatter(x[tr], y[tr], s=30, color=TEAL, label="entraînement")
        xs = np.linspace(x[tr].min(), x[tr].max(), 300)
        ax1.plot(xs, models[d].predict(xs[:, None]), color=ANIS, lw=3)
        ax1.set_ylim(-2000, np.percentile(y, 99.5) * 1.1)
        ax1.set_title(f"degré {d}")
        ax1.set_xlabel("carat")
        ax1.set_ylabel("prix ($)")
        ax1.legend(loc="upper left")
        ax2.plot(degrees, err_tr, "o-", color=TEAL, label="erreur entraînement")
        ax2.plot(degrees, err_te, "o-", color=RED, label="erreur test")
        ax2.axvline(d, color=ANIS, lw=2)
        ax2.set_yscale("log")
        ax2.set_xlabel("degré du polynôme")
        ax2.set_ylabel("MAE ($, échelle log)")
        ax2.legend()
        plt.tight_layout()
        plt.show()

    output = w.interactive_output(draw, {"d": slider})
    display(slider, output)


def ridge_slider(x_train, y_train, x_val=None, y_val=None, degree: int = 12):
    """Un curseur alpha (Ridge) sur un polynôme de degré élevé : les coefficients s'écrasent, la courbe se calme."""
    import warnings

    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

    x_train = np.asarray(x_train, dtype=float).ravel()
    y_train = np.asarray(y_train, dtype=float)
    if x_val is not None:
        x_val = np.asarray(x_val, dtype=float).ravel()
        y_val = np.asarray(y_val, dtype=float)

    alpha = w.FloatLogSlider(value=1e-6, base=10, min=-6, max=2, step=0.25, description="alpha",
                             readout_format=".0e", style={"description_width": "70px"}, layout=w.Layout(width="420px"))

    def draw(a):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            regression = Ridge(alpha=a) if a > 1e-6 else LinearRegression()
            m = make_pipeline(MinMaxScaler((-1, 1)), PolynomialFeatures(degree), regression).fit(x_train[:, None], y_train)
        coefs = m[-1].coef_[1:]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        if x_val is not None:
            ax1.scatter(x_val, y_val, s=8, alpha=0.3, color=GREY, label="validation")
        ax1.scatter(x_train, y_train, s=30, color=TEAL, label="entraînement")
        xs = np.linspace(x_train.min(), x_train.max(), 300)
        ax1.plot(xs, m.predict(xs[:, None]), color=ANIS, lw=3)
        ax1.set_ylim(-2000, max(y_train) * 1.4)
        title = "sans Ridge" if a <= 1e-6 else f"Ridge, alpha = {a:.0e}"
        if x_val is not None:
            mae = np.mean(np.abs(m.predict(x_val[:, None]) - y_val))
            title += f"   ·   MAE validation = {mae:,.0f} $".replace(",", " ")
        ax1.set_title(title)
        ax1.set_xlabel("carat")
        ax1.set_ylabel("prix ($)")
        ax1.legend(loc="upper left")
        ax2.bar(range(1, degree + 1), np.abs(coefs), color=[RED if c < 0 else TEAL for c in coefs])
        ax2.set_yscale("log")
        ax2.set_ylim(1, 1e9)
        ax2.set_xlabel("puissance de carat (1 = carat, 2 = carat², …)")
        ax2.set_ylabel("|coefficient| (échelle log)")
        ax2.set_title("taille des coefficients (rouge = négatif)")
        plt.tight_layout()
        plt.show()

    output = w.interactive_output(draw, {"a": alpha})
    display(alpha, output)


def _grid(X, n=200, margin=0.3):
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    dx, dy = (x_max - x_min) * margin / 2, (y_max - y_min) * margin / 2
    xx, yy = np.meshgrid(np.linspace(x_min - dx, x_max + dx, n), np.linspace(y_min - dy, y_max + dy, n))
    return xx, yy


def knn_boundary(n_points: int = 300, noise: float = 0.3, seed: int = 0):
    """Un curseur k sur des données en forme de lunes : frontière déchiquetée à k = 1, trop lisse à k = 150."""
    from matplotlib.colors import ListedColormap
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier

    X, y = make_moons(n_samples=n_points * 2, noise=noise, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.5, random_state=seed)
    xx, yy = _grid(X)
    background = ListedColormap(["#C8E6E6", "#EEF2C7"])
    slider = w.SelectionSlider(options=[1, 2, 3, 5, 10, 15, 25, 50, 100, 150], value=1, description="k voisins",
                                style={"description_width": "70px"}, layout=w.Layout(width="420px"))

    def draw(k):
        m = KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr)
        zz = m.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.contourf(xx, yy, zz, cmap=background, alpha=0.9)
        ax.contour(xx, yy, zz, levels=[0.5], colors=GREY, linewidths=1.5)
        ax.scatter(X_tr[y_tr == 0, 0], X_tr[y_tr == 0, 1], s=18, color=TEAL, label="classe A")
        ax.scatter(X_tr[y_tr == 1, 0], X_tr[y_tr == 1, 1], s=18, color=ANIS, edgecolor=GREY, label="classe B")
        acc_tr, acc_va = m.score(X_tr, y_tr), m.score(X_va, y_va)
        ax.set_title(f"k = {k}   ·   exactitude entraînement {acc_tr:.0%}   ·   validation {acc_va:.0%}", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="upper right")
        plt.show()

    output = w.interactive_output(draw, {"k": slider})
    display(slider, output)


def tree_depth(df, x="vitesse", y="couple", target="panne", seed: int = 0):
    """Un curseur de profondeur pour un arbre de décision sur deux variables (par défaut vitesse et couple d'AI4I)."""
    from matplotlib.colors import ListedColormap
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    X = df[[x, y]].to_numpy(dtype=float)
    target_values = df[target].to_numpy()
    X_tr, X_va, y_tr, y_va = train_test_split(X, target_values, test_size=0.3, random_state=seed, stratify=target_values)
    xx, yy = _grid(X, margin=0.05)
    background = ListedColormap(["#E0F0F0", "#F1D4D0"])
    slider = w.IntSlider(value=1, min=1, max=15, description="profondeur",
                          style={"description_width": "70px"}, layout=w.Layout(width="420px"))
    depths = list(range(1, 16))
    f1_tr, f1_va = [], []
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_tr, y_tr)
        f1_tr.append(f1_score(y_tr, m.predict(X_tr)))
        f1_va.append(f1_score(y_va, m.predict(X_va)))

    def draw(d):
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_tr, y_tr)
        zz = m.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.4, 1]})
        ax1.contourf(xx, yy, zz, cmap=background, alpha=0.9)
        ok = y_tr == 0
        ax1.scatter(X_tr[ok, 0], X_tr[ok, 1], s=4, color=TEAL, alpha=0.25, label="pas de panne")
        ax1.scatter(X_tr[~ok, 0], X_tr[~ok, 1], s=22, color=RED, label="panne")
        ax1.set_xlabel(x)
        ax1.set_ylabel(y)
        ax1.set_title(f"profondeur {d} : {m.get_n_leaves()} feuilles (zones rouges = « panne » prédite)", fontsize=11)
        ax1.legend(loc="upper right")
        ax2.plot(depths, f1_tr, "o-", color=TEAL, label="F1 entraînement")
        ax2.plot(depths, f1_va, "o-", color=RED, label="F1 validation")
        ax2.axvline(d, color=ANIS, lw=2)
        ax2.set_ylim(0, 1.05)
        ax2.set_xlabel("profondeur de l'arbre")
        ax2.set_ylabel("F1 (pannes)")
        ax2.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

    output = w.interactive_output(draw, {"d": slider})
    display(slider, output)


def kmeans_step_by_step(n_points: int = 300, true_clusters: int = 4):
    """k-means étape par étape sur des nuages de points : curseurs k, seed (le départ) et étape."""
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=n_points, centers=true_clusters, cluster_std=1.1, random_state=7)
    colors = [TEAL, ANIS, RED, "#2799D7", "#7C5BA6", "#E08E2B", GREY, "#0A5F5F"]
    k_w = w.IntSlider(value=4, min=2, max=8, description="k", style={"description_width": "70px"}, layout=w.Layout(width="420px"))
    seed_w = w.IntSlider(value=0, min=0, max=20, description="départ", style={"description_width": "70px"},
                           layout=w.Layout(width="420px"))
    step_w = w.IntSlider(value=0, min=0, max=10, description="étape", style={"description_width": "70px"},
                          layout=w.Layout(width="420px"))

    def draw(k, g, step):
        rng = np.random.default_rng(g)
        centers = X[rng.choice(len(X), k, replace=False)].copy()
        labels = np.zeros(len(X), dtype=int)
        for _ in range(step):
            distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)
            for j in range(k):
                if np.any(labels == j):
                    centers[j] = X[labels == j].mean(axis=0)
        if step > 0:
            distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)
        inertia = ((X - centers[labels]) ** 2).sum()
        fig, ax = plt.subplots(figsize=(7.5, 5))
        if step == 0:
            ax.scatter(X[:, 0], X[:, 1], s=14, color=GREY, alpha=0.6)
        else:
            for j in range(k):
                ax.scatter(X[labels == j, 0], X[labels == j, 1], s=14, color=colors[j % len(colors)], alpha=0.6)
        ax.scatter(centers[:, 0], centers[:, 1], s=320, marker="X", color="black", edgecolor="white", linewidth=1.5)
        title = "étape 0 : centres tirés au hasard" if step == 0 else f"étape {step}   ·   inertie = {inertia:,.0f}".replace(",", " ")
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

    output = w.interactive_output(draw, {"k": k_w, "g": seed_w, "step": step_w})
    display(w.VBox([k_w, seed_w, step_w]), output)
