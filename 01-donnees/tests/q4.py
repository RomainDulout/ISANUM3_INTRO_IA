from otter.test_files import test_case

import course_tools as ct

OK_FORMAT = False
name = "q4"

VALID_CUTS = {"Fair", "Good", "Very Good", "Premium", "Ideal"}


def reference():
    df = ct.load_dirty_diamonds().drop_duplicates()
    df["cut"] = df["cut"].str.strip().replace({"Premuim": "Premium", "ideal": "Ideal"})
    df = df.dropna(subset=["price"])
    df.loc[df["carat"] > 10, "carat"] = df.loc[df["carat"] > 10, "carat"] / 200
    df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0) & (df["y"] < 30) & (df["z"] < 30)]
    for column in ["carat", "depth", "table"]:
        df[column] = df[column].fillna(df[column].median())
    df["color"] = df["color"].fillna(df["color"].mode()[0])
    return df


@test_case(points=1, success_message="✅ a. Plus aucun doublon.")
def test_duplicates(df_clean):
    assert df_clean.duplicated().sum() == 0, "❌ a. Il reste des doublons : avez-vous bien écrit df_clean = df_clean.drop_duplicates() ?"


@test_case(points=1, success_message="✅ b. La colonne cut ne contient plus que les 5 tailles valides.")
def test_cut(df_clean):
    remaining = set(df_clean["cut"].unique()) - VALID_CUTS
    assert not remaining, (
        f"❌ b. Il reste des valeurs à corriger dans cut : {sorted(remaining)}. "
        "Le dictionnaire de replace doit associer chaque faute à la bonne orthographe."
    )


@test_case(points=1, success_message="✅ c. Plus aucun prix manquant.")
def test_price(df_clean):
    assert df_clean["price"].isnull().sum() == 0, "❌ c. Il reste des prix manquants : utilisez df_clean.dropna(subset=[\"price\"])."


@test_case(points=1, success_message="✅ d. Poids et dimensions plausibles.")
def test_dimensions(df_clean):
    assert df_clean["carat"].max() < 10, "❌ d. Il reste des poids en milligrammes : divisez par 200 les carats supérieurs à 10."
    assert (df_clean[["x", "y", "z"]] > 0).all().all(), "❌ d. Il reste des dimensions nulles : les trois conditions x > 0, y > 0 et z > 0 sont nécessaires."
    assert (df_clean["y"] < 30).all() and (df_clean["z"] < 30).all(), "❌ d. Il reste des dimensions aberrantes : ajoutez les conditions y < 30 et z < 30."


@test_case(points=1, success_message="✅ e. Plus aucune valeur manquante.")
def test_imputation(df_clean):
    remaining = df_clean.isnull().sum()
    remaining = remaining[remaining > 0]
    assert remaining.empty, f"❌ e. Il reste des valeurs manquantes dans : {list(remaining.index)}."


@test_case(points=1, success_message="✅ Le tableau final correspond à la recette.")
def test_final(df_clean):
    ref = reference()
    assert len(df_clean) == len(ref), (
        "❌ Le nombre de lignes ne correspond pas à la recette. Vérifiez l'ordre des étapes a → e "
        "et repars de la cellule df_clean = df.copy() si besoin."
    )
    gap = (df_clean["carat"].sort_values().to_numpy() - ref["carat"].sort_values().to_numpy())
    assert abs(gap).max() < 1e-9, "❌ Les valeurs de carat ne correspondent pas : avez-vous imputé avec la médiane (et non la moyenne) ?"
