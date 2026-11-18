import hashlib

from otter.test_files import test_case

import course_tools as ct

OK_FORMAT = False
name = "q6"

EXPECTED_HASH = "6b23c0d5f35d1b11f9b683f0b0a617355deb11277d91ae091d399c655b87940d"


@test_case(points=1, success_message="✅ Les deux scores sont justes.")
def test_scores(r2_train, r2_test):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    X_train, X_test, y_train, y_test = ct._split_diamonds()
    model = LinearRegression().fit(X_train, y_train)
    expected_train = r2_score(y_train, model.predict(X_train))
    expected_test = r2_score(y_test, model.predict(X_test))
    assert r2_train is not ... and r2_test is not ..., "❌ Remplacez les ... de r2_train et r2_test."
    assert abs(float(r2_train) - expected_train) < 1e-3, (
        "❌ r2_train : comparez y_train aux prédictions du modèle sur X_train : r2_score(y_train, multiple_model.predict(X_train))."
    )
    assert abs(float(r2_test) - expected_test) < 1e-3, "❌ r2_test : r2_score(y_test, multiple_model.predict(X_test))."


@test_case(points=1, success_message="✅ Juste : une droite est trop simple pour apprendre par cœur 4 000 diamants.")
def test_question(answer_generalization):
    answer = str(answer_generalization).strip().upper()
    assert answer in {"A", "B", "C", "D"}, "❌ Répondez par une lettre entre guillemets : \"A\", \"B\", \"C\" ou \"D\"."
    assert answer != "A", "❌ Le R² d'entraînement est un peu plus élevé, mais l'écart est minime : on parle de surapprentissage quand l'écart est net."
    assert answer != "B", "❌ Un R² d'environ 0,88 est bon, mais pas parfait : il reste environ 900 $ d'erreur moyenne."
    assert answer != "D", "❌ Le R² suffit pour comparer entraînement et test : ce qui compte, c'est l'écart entre les deux."
    assert hashlib.sha256(answer.encode()).hexdigest() == EXPECTED_HASH, "❌ Relisez les propositions."
