from otter.test_files import test_case

import course_tools as ct

OK_FORMAT = False
name = "q3"


@test_case(points=1, success_message="✅ La descente arrive au même endroit que scikit-learn : c'est un apprentissage codé à la main.")
def test_descent(a_gd, b_gd):
    import numpy as np
    from sklearn.linear_model import LinearRegression

    X_train, X_test, y_train, y_test = ct._split_diamonds()
    model = LinearRegression().fit(X_train[["carat"]], y_train)
    a_expected, b_expected = model.coef_[0], model.intercept_
    assert a_gd is not ... and b_gd is not ..., "❌ Remplacez les ... de la boucle par vos calculs."
    assert np.isfinite(a_gd) and np.isfinite(b_gd), (
        "❌ a ou b vaut inf ou nan : la descente a divergé. Avez-vous gardé eta = 0.1 ? "
        "Vérifiez le signe de vos dérivées partielles et le signe moins de la mise à jour."
    )
    assert a_gd != 0 and b_gd != 0, "❌ a ou b n'a pas bougé : mettez à jour les deux paramètres à chaque itération."
    assert abs(a_gd - a_expected) / abs(a_expected) < 0.01 and abs(b_gd - b_expected) / abs(b_expected) < 0.01, (
        "❌ La descente n'arrive pas au même résultat que scikit-learn. Vérifiez vos deux dérivées partielles "
        "(le facteur 2, le signe, et le x_i dans celle par rapport à a) et la règle a ← a - η × ∂J/∂a."
    )


@test_case(points=0, success_message="✅ L'historique de la MSE est rempli à chaque itération.")
def test_history(history):
    import numpy as np

    assert len(history) == 3000, "❌ history doit contenir une MSE par itération (3 000 au total) : history.append(...) dans la boucle."
    values = np.asarray(history, dtype=float)
    assert values[-1] < values[0], "❌ La MSE devrait baisser au fil des itérations : ajoutez à history la moyenne des erreurs au carré."
