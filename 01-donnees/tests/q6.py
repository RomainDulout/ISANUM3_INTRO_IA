import numpy as np
from otter.test_files import test_case

OK_FORMAT = False
name = "q6"


@test_case(points=1, success_message="✅ X et y sont bien séparés.")
def test_xy(X, y):
    assert "price" not in X.columns, "❌ X ne doit pas contenir la cible : X = df_enc.drop(columns=\"price\")."
    assert y.name == "price", "❌ y doit être la colonne price : y = df_enc[\"price\"]."


@test_case(points=1, success_message="✅ Découpage 80 % / 20 % correct.")
def test_split(X, X_train, X_test):
    assert len(X_train) + len(X_test) == len(X), "❌ X_train et X_test doivent contenir à eux deux toutes les lignes de X."
    assert abs(len(X_test) / len(X) - 0.2) < 0.01, "❌ Le test doit représenter 20 % des données : test_size=0.2."


@test_case(points=1, success_message="✅ Normalisation correcte : le scaler a appris sur l'entraînement uniquement.")
def test_scaler(X_train, X_test, X_train_sc, X_test_sc):
    assert X_train_sc.shape == X_train.shape and X_test_sc.shape == X_test.shape, (
        "❌ X_train_sc doit venir de X_train, et X_test_sc de X_test."
    )
    assert np.allclose(X_train_sc.mean(axis=0), 0, atol=1e-6), "❌ X_train_sc doit avoir une moyenne de 0 : utilisez fit_transform(X_train)."
    expected = (X_test.to_numpy() - X_train.to_numpy().mean(axis=0)) / X_train.to_numpy().std(axis=0)
    assert np.allclose(X_test_sc, expected, atol=1e-6), (
        "❌ X_test_sc doit être transformé avec les moyennes et écarts-types de l'entraînement : "
        "scaler.transform(X_test), surtout pas fit_transform."
    )
