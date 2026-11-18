from otter.test_files import test_case

import course_tools as ct

OK_FORMAT = False
name = "bonus3"


@test_case(points=1, success_message="✅ Bonus 3 réussi : l'erreur est divisée par deux avec une simple régression linéaire.")
def test_bonus3(predictions_bonus):
    import numpy as np
    from sklearn.linear_model import LinearRegression

    X_train, X_test, y_train, y_test = ct._split_diamonds()
    mae_multiple = np.mean(np.abs(y_test - LinearRegression().fit(X_train, y_train).predict(X_test)))
    assert predictions_bonus is not ..., "❌ Remplacez les ... de predictions_bonus."
    pred = np.asarray(predictions_bonus, dtype=float).ravel()
    assert len(pred) == len(y_test), f"❌ Il faut une prédiction par diamant du test ({len(y_test)}), vous en avez {len(pred)}."
    assert np.all(np.isfinite(pred)), "❌ Certaines prédictions valent inf ou nan."
    assert np.median(pred) > 100, "❌ Vos prédictions ne sont pas en dollars : si vous avez prédit log(prix), repassez en dollars avec np.exp."
    mae = np.mean(np.abs(y_test.to_numpy() - pred))
    assert mae <= mae_multiple / 2, (
        f"❌ MAE = {mae:.0f} $, il faut descendre à {mae_multiple / 2:.0f} $. Ouvrez les indices si vous bloquez."
    )
