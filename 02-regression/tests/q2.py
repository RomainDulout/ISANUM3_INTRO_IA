from otter.test_files import test_case

import course_tools as ct

OK_FORMAT = False
name = "q2"


def _expected():
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error

    X_train, X_test, y_train, y_test = ct._split_diamonds()
    model = LinearRegression().fit(X_train[["carat"]], y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test[["carat"]]))
    return model.coef_[0], model.intercept_, mae


@test_case(points=1, success_message="✅ Le modèle a bien appris sa pente et son ordonnée.")
def test_parameters(slope, intercept):
    expected_slope, expected_intercept, _ = _expected()
    assert slope is not ... and intercept is not ..., "❌ Remplacez les ... de slope et intercept."
    assert abs(float(slope) - expected_slope) < 1, (
        "❌ slope : avez-vous entraîné le modèle sur la seule colonne carat (avec deux paires de crochets) ? "
        "La pente est le premier élément de simple_model.coef_."
    )
    assert abs(float(intercept) - expected_intercept) < 1, "❌ intercept : l'ordonnée à l'origine est dans simple_model.intercept_."


@test_case(points=1, success_message="✅ Un prix négatif : le modèle extrapole hors de ses données.")
def test_zero_carat(price_0_carat):
    _, expected_intercept, _ = _expected()
    assert price_0_carat is not ..., "❌ Remplacez les ... de price_0_carat."
    assert abs(float(price_0_carat) - expected_intercept) < 1, (
        "❌ price_0_carat : dans prix = a × carat + b, que vaut le prix quand carat = 0 ?"
    )


@test_case(points=1, success_message="✅ L'erreur est divisée par presque trois par rapport au modèle de référence.")
def test_mae_simple(mae_simple):
    _, _, expected = _expected()
    assert mae_simple is not ..., "❌ Remplacez les ... de mae_simple."
    assert abs(float(mae_simple) - expected) < 1, (
        "❌ mae_simple : utilisez mean_absolute_error(y_test, predictions_simple), dans cet ordre et sur le test."
    )
