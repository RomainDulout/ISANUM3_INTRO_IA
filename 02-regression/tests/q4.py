from otter.test_files import test_case

import course_tools as ct

OK_FORMAT = False
name = "q4"


@test_case(points=1, success_message="✅ Mieux que le modèle simple, mais le gain est plus modeste qu'on pourrait l'espérer.")
def test_multiple(mae_multiple, predictions):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error

    X_train, X_test, y_train, y_test = ct._split_diamonds()
    model = LinearRegression().fit(X_train, y_train)
    expected = mean_absolute_error(y_test, model.predict(X_test))
    simple = mean_absolute_error(y_test, LinearRegression().fit(X_train[["carat"]], y_train).predict(X_test[["carat"]]))
    assert mae_multiple is not ..., "❌ Remplacez les ... de mae_multiple."
    assert len(predictions) == len(y_test), "❌ predictions doit contenir un prix prédit pour chaque diamant du test (X_test)."
    assert abs(float(mae_multiple) - simple) > 1, (
        "❌ Vous obtenez la même erreur que le modèle simple : entraînez sur toutes les colonnes de X_train, pas seulement sur carat."
    )
    assert abs(float(mae_multiple) - expected) < 1, (
        "❌ mae_multiple : entraînez sur toutes les variables d'entraînement, prédisez sur X_test, "
        "puis calculez la MAE avec mean_absolute_error."
    )
