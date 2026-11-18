import hashlib

from otter.test_files import test_case

import course_tools as ct

OK_FORMAT = False
name = "q5"

EXPECTED_HASH = "df7e70e5021544f4834bbee64a9e3789febc4be81470df629cad6ddb03320a5c"


@test_case(points=1, success_message="✅ MAE, RMSE et R² sont justes.")
def test_metrics(mae, rmse, r2):
    import numpy as np
    from sklearn.linear_model import LinearRegression

    X_train, X_test, y_train, y_test = ct._split_diamonds()
    pred = LinearRegression().fit(X_train, y_train).predict(X_test)
    expected_rmse = np.sqrt(np.mean((y_test - pred) ** 2))
    expected_r2 = 1 - np.mean((y_test - pred) ** 2) / np.mean((y_test - y_test.mean()) ** 2)
    assert mae is not ... and rmse is not ... and r2 is not ..., "❌ Remplacez les ... de mae, rmse et r2."
    assert abs(float(mae) - np.mean(np.abs(y_test - pred))) < 1, (
        "❌ mae : la moyenne des valeurs absolues des erreurs, avec mean_absolute_error ou np.mean et np.abs."
    )
    assert abs(float(rmse) - expected_rmse ** 2) > 1, "❌ rmse : vous avez calculé la MSE. Il manque la racine carrée."
    assert abs(float(rmse) - expected_rmse) < 1, "❌ rmse : la racine de la moyenne des erreurs au carré, avec root_mean_squared_error ou np.sqrt et np.mean."
    assert abs(float(r2) - expected_r2) < 1e-3, (
        "❌ r2 : avec r2_score, les vrais prix d'abord ; à la main, 1 - MSE du modèle / MSE obtenue en prédisant "
        "la moyenne de y_test pour tous."
    )


@test_case(points=1, success_message="✅ Juste : la RMSE punit fort les grosses erreurs, souvent sur les diamants les plus chers.")
def test_question(answer_rmse):
    answer = str(answer_rmse).strip().upper()
    assert answer in {"A", "B", "C", "D"}, "❌ Répondez par une lettre entre guillemets : \"A\", \"B\", \"C\" ou \"D\"."
    assert answer != "A", "❌ La MAE prend la valeur absolue : elle compte aussi les erreurs dans les deux sens."
    assert answer != "C", "❌ Grâce à la racine carrée, la RMSE est bien en dollars, comme la MAE."
    assert answer != "D", "❌ C'est l'inverse : mettre au carré écrase les petites erreurs (0,5² = 0,25) et gonfle les grosses."
    assert hashlib.sha256(answer.encode()).hexdigest() == EXPECTED_HASH, "❌ Relisez les propositions."
