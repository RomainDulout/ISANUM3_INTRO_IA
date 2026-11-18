from otter.test_files import test_case

import course_tools as ct

OK_FORMAT = False
name = "bonus1"


@test_case(points=1, success_message="✅ Les résidus sont justes.")
def test_residuals(residuals):
    import numpy as np
    from sklearn.linear_model import LinearRegression

    X_train, X_test, y_train, y_test = ct._split_diamonds()
    pred = LinearRegression().fit(X_train, y_train).predict(X_test)
    assert residuals is not ..., "❌ Remplacez les ... de residuals."
    assert len(residuals) == len(y_test), "❌ residuals doit contenir un résidu par diamant du test."
    values = np.asarray(residuals, dtype=float)
    if np.allclose(values, pred - np.asarray(y_test), atol=1):
        raise AssertionError("❌ Le signe est inversé : un résidu, c'est vrai prix - prix prédit.")
    assert np.allclose(values, np.asarray(y_test) - pred, atol=1), (
        "❌ residuals : pour chaque diamant du test, le vrai prix moins le prix prédit par le modèle multiple."
    )


@test_case(points=1, success_message="✅ Des prix négatifs : impossible pour un vrai diamant.")
def test_negatives(n_negative_prices):
    from sklearn.linear_model import LinearRegression

    X_train, X_test, y_train, y_test = ct._split_diamonds()
    pred = LinearRegression().fit(X_train, y_train).predict(X_test)
    assert n_negative_prices is not ..., "❌ Remplacez les ... de n_negative_prices."
    assert int(n_negative_prices) == int((pred < 0).sum()), (
        "❌ n_negative_prices : comptez les prédictions inférieures à 0 (une comparaison donne des True / False, et .sum() compte les True)."
    )


@test_case(points=1, success_message="✅ Le logarithme fait gagner beaucoup, sans changer d'algorithme.")
def test_log(mae_log):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error

    X_train, X_test, y_train, y_test = ct._split_diamonds()
    mae_multiple = mean_absolute_error(y_test, LinearRegression().fit(X_train, y_train).predict(X_test))
    assert mae_log is not ..., "❌ Remplacez les ... de mae_log."
    assert float(mae_log) > 50, (
        "❌ mae_log est bien trop petite : avez-vous comparé des log(prix) entre eux ? Repassez en dollars avec np.exp avant de calculer la MAE."
    )
    assert float(mae_log) < 0.8 * mae_multiple, (
        "❌ mae_log n'est pas meilleure que le modèle multiple. Vérifiez les quatre étapes : entraîner sur np.log des prix, "
        "prédire sur X_test, repasser en dollars avec np.exp, puis la MAE avec les vrais prix y_test."
    )
