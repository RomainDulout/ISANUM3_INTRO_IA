from otter.test_files import test_case

import course_tools as ct

OK_FORMAT = False
name = "q1"


@test_case(points=1, success_message="✅ Voilà la barre à franchir : un vrai modèle doit faire nettement mieux.")
def test_baseline(mae_baseline):
    import numpy as np

    X_train, X_test, y_train, y_test = ct._split_diamonds()
    expected = np.mean(np.abs(y_test - y_train.mean()))
    assert mae_baseline is not ..., "❌ mae_baseline : remplacez les ... par votre calcul."
    assert np.isscalar(mae_baseline) or np.ndim(mae_baseline) == 0, (
        "❌ mae_baseline doit être un seul nombre : n'oubliez pas np.mean(...) autour des écarts."
    )
    assert float(mae_baseline) >= 0, "❌ Une MAE ne peut pas être négative : avez-vous pris la valeur absolue avec np.abs ?"
    if abs(float(mae_baseline) - np.mean(np.abs(y_train - y_train.mean()))) < 1e-6:
        raise AssertionError("❌ Vous avez calculé l'erreur sur l'entraînement : il faut comparer y_test et predictions_baseline.")
    assert abs(float(mae_baseline) - expected) < 1, (
        "❌ mae_baseline : calculez la moyenne de |y_test - predictions_baseline|, "
        "c'est-à-dire np.mean(np.abs(y_test - predictions_baseline))."
    )
