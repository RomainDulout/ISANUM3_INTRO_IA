"""Petits outils du cours R5-07 Introduction à l'IA.

Les étudiants n'ont pas besoin de lire ce fichier. Il sert à :

- `start(first_name)` : tirer une seed personnelle à partir du prénom,
  pour que chaque étudiant travaille sur son propre échantillon de données ;
- `load_...()` : charger les jeux de données du cours depuis le dossier
  `donnees/`, sans connexion internet ;
- `check_install()` : vérifier que l'environnement est bien installé.
"""

from __future__ import annotations

import hashlib
import platform
import sys
import unicodedata
from pathlib import Path

import pandas as pd

_SEED: int | None = None
_FIRST_NAME: str | None = None


def _root() -> Path:
    """Dossier racine du cours (celui qui contient pyproject.toml)."""
    for folder in [Path.cwd(), *Path.cwd().parents]:
        if (folder / "pyproject.toml").exists() and (folder / "donnees").exists():
            return folder
    return Path(__file__).resolve().parents[1]


def _normalize(first_name: str) -> str:
    text = unicodedata.normalize("NFKD", first_name.strip().lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def start(first_name: str) -> int:
    """Enregistre le prénom et calcule la seed personnelle (toujours la même pour un prénom)."""
    global _SEED, _FIRST_NAME
    if not isinstance(first_name, str) or not first_name.strip() or first_name.strip() in {"...", "Ellipsis"}:
        raise ValueError("❌ Écrivez votre prénom entre guillemets, par exemple : ct.start(\"Camille\")")
    _FIRST_NAME = first_name.strip()
    digest = hashlib.sha256(_normalize(first_name).encode()).hexdigest()
    _SEED = int(digest[:8], 16) % 100_000
    print(f"Bonjour {_FIRST_NAME}. Seed personnelle : {_SEED}.")
    print("Elle fixe le hasard des notebooks (tirage des données, départ des algorithmes) : les résultats varient un peu d'une personne à l'autre.")
    return _SEED


def seed() -> int:
    """Renvoie la seed personnelle (il faut avoir appelé start avant)."""
    if _SEED is None:
        raise RuntimeError("❌ Exécutez d'abord la cellule ct.start(\"VotrePrénom\") en haut du notebook.")
    return _SEED


def load_dirty_diamonds(n: int = 5000) -> pd.DataFrame:
    """Échantillon personnel de n diamants tirés du fichier « sali » (avec des défauts à nettoyer)."""
    df = pd.read_csv(_root() / "donnees" / "diamonds_sali.csv")
    return df.sample(n=n, random_state=seed()).reset_index(drop=True)


CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]


def load_clean_diamonds() -> pd.DataFrame:
    """Échantillon personnel, nettoyé et encodé comme dans la correction du NB01.

    Permet de repartir d'une base propre même si le NB01 n'est pas terminé.
    """
    df = load_dirty_diamonds().drop_duplicates()
    df["cut"] = df["cut"].str.strip().replace({"Premuim": "Premium", "ideal": "Ideal"})
    df = df.dropna(subset=["price"])
    df.loc[df["carat"] > 10, "carat"] = df.loc[df["carat"] > 10, "carat"] / 200
    df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0) & (df["y"] < 30) & (df["z"] < 30)]
    for column in ["carat", "depth", "table"]:
        df[column] = df[column].fillna(df[column].median())
    df["color"] = df["color"].fillna(df["color"].mode()[0])
    for column, order in [("cut", CUT_ORDER), ("color", COLOR_ORDER), ("clarity", CLARITY_ORDER)]:
        df[column] = df[column].map({value: rank for rank, value in enumerate(order)})
    return df.reset_index(drop=True)


def _split_diamonds():
    """Découpage train / test du NB02, identique à celui du notebook (sert aux vérifications)."""
    from sklearn.model_selection import train_test_split

    df = load_clean_diamonds()
    X = df.drop(columns="price")
    y = df["price"]
    return train_test_split(X, y, test_size=0.2, random_state=seed())


def _small_sets():
    """Les trois jeux du NB03 (30 diamants d'entraînement, 200 de validation, le reste en test)."""
    df = load_clean_diamonds()
    training = df.sample(n=30, random_state=seed())
    rest = df.drop(training.index)
    rest = rest[(rest["carat"] >= training["carat"].min()) & (rest["carat"] <= training["carat"].max())]
    validation = rest.sample(n=200, random_state=seed())
    test = rest.drop(validation.index)
    return training, validation, test


def load_wholesale_raw() -> pd.DataFrame:
    """Dépenses annuelles de 440 clients d'un grossiste (UCI Wholesale Customers)."""
    return pd.read_csv(_root() / "donnees" / "wholesale_customers.csv")


WHOLESALE_COLUMNS = {
    "Channel": "canal", "Region": "region", "Fresh": "frais", "Milk": "laitier", "Grocery": "epicerie",
    "Frozen": "surgeles", "Detergents_Paper": "droguerie", "Delicassen": "traiteur",
}
SPENDING = ["frais", "laitier", "epicerie", "surgeles", "droguerie", "traiteur"]


def load_wholesale() -> pd.DataFrame:
    """Wholesale Customers avec des noms de colonnes en français.

    canal : 1 = hôtels, restaurants, cafés ; 2 = commerce de détail. region : 1 = Lisbonne, 2 = Porto, 3 = autre.
    """
    return load_wholesale_raw().rename(columns=WHOLESALE_COLUMNS)


def _wholesale_scaled():
    """Dépenses passées au log puis normalisées, comme dans le NB05 (sert aux vérifications)."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    return StandardScaler().fit_transform(np.log(load_wholesale()[SPENDING]))


def load_ai4i_raw() -> pd.DataFrame:
    """10 000 cycles d'une machine-outil, dont ~3,4 % de pannes (AI4I 2020, S. Matzka, UCI)."""
    return pd.read_csv(_root() / "donnees" / "ai4i2020.csv", encoding="utf-8-sig")


AI4I_COLUMNS = {
    "Air temperature [K]": "temp_air",
    "Process temperature [K]": "temp_process",
    "Rotational speed [rpm]": "vitesse",
    "Torque [Nm]": "couple",
    "Tool wear [min]": "usure",
    "Machine failure": "panne",
    "TWF": "panne_usure",
    "HDF": "panne_chaleur",
    "PWF": "panne_puissance",
    "OSF": "panne_surcharge",
    "RNF": "panne_aleatoire",
}
AI4I_FEATURES = ["type", "temp_air", "temp_process", "vitesse", "couple", "usure"]


def load_failures() -> pd.DataFrame:
    """AI4I 2020 avec des noms de colonnes en français et le type de produit encodé (L = 0, M = 1, H = 2)."""
    df = load_ai4i_raw().rename(columns=AI4I_COLUMNS)
    df["type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})
    columns = AI4I_FEATURES + ["panne", "panne_usure", "panne_chaleur", "panne_puissance", "panne_surcharge", "panne_aleatoire"]
    return df[columns]


def _split_failures():
    """Découpage 60 / 20 / 20 du NB04 (entraînement, validation, test), stratifié sur la panne."""
    from sklearn.model_selection import train_test_split

    df = load_failures()
    X, y = df[AI4I_FEATURES], df["panne"]
    X_rest, X_test, y_rest, y_test = train_test_split(X, y, test_size=0.2, random_state=seed(), stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=0.25, random_state=seed(), stratify=y_rest
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_ames() -> pd.DataFrame:
    """2 930 maisons vendues à Ames (Iowa), 82 colonnes (Ames Housing, D. De Cock)."""
    return pd.read_csv(_root() / "donnees" / "ames_housing.csv")


def personal_folder() -> Path:
    """Dossier où chacun sauvegarde ses fichiers de travail (non partagé, ignoré par git)."""
    folder = _root() / "donnees_perso"
    folder.mkdir(exist_ok=True)
    return folder


def check_install() -> None:
    """Affiche les versions des bibliothèques du cours et signale ce qui manque."""
    print(f"Python {sys.version.split()[0]} sur {platform.system()} {platform.machine()}")
    ok = True
    for name in ["pandas", "numpy", "matplotlib", "seaborn", "sklearn", "ipywidgets", "otter"]:
        try:
            module = __import__(name)
            print(f"  ✅ {name} {getattr(module, '__version__', '')}")
        except ImportError:
            ok = False
            print(f"  ❌ {name} manquant : dans le terminal, lancez « uv sync » dans le dossier du cours")
    try:
        import torch

        mps = torch.backends.mps.is_available()
        print(f"  ✅ torch {torch.__version__} (accélération Apple mps : {'oui' if mps else 'non'})")
    except ImportError:
        print("  (info) torch pas encore installé : normal, il ne sert que pour les réseaux de neurones.")
    print("Tout est prêt." if ok else "Il manque quelque chose : levez la main pour demander de l'aide.")
