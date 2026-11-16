from otter.test_files import test_case

import course_tools as ct

OK_FORMAT = False
name = "q3"

VALID_CUTS = ["Fair", "Good", "Very Good", "Premium", "Ideal"]


@test_case(points=1, success_message="✅ Juste d'après la heatmap. La valeur de carat est pourtant étonnamment faible.")
def test_correlation(most_correlated_variable):
    df = ct.load_dirty_diamonds()
    correlations = df.select_dtypes("number").corr()["price"].drop("price")
    expected = correlations.idxmax()
    assert isinstance(most_correlated_variable, str) and most_correlated_variable != "...", (
        "❌ most_correlated_variable : écrivez le nom de la colonne entre guillemets, par exemple \"depth\"."
    )
    assert most_correlated_variable.strip() == expected, (
        "❌ most_correlated_variable : dans la heatmap, regardez la ligne « price » et cherchez la plus grande valeur "
        "(hors price elle-même). Attention, ce n'est pas forcément celle que vous attendiez !"
    )


@test_case(points=1, success_message="✅ Juste, et c'est surprenant : voir la question qui suit.")
def test_cut(cheapest_cut):
    df = ct.load_dirty_diamonds()
    expected = df[df["cut"].isin(VALID_CUTS)].groupby("cut")["price"].mean().idxmin()
    assert isinstance(cheapest_cut, str) and cheapest_cut in VALID_CUTS, (
        "❌ cheapest_cut : écrivez une des 5 tailles valides entre guillemets, par exemple \"Good\"."
    )
    assert cheapest_cut == expected, (
        "❌ cheapest_cut : regardez le résultat du groupby, trié du moins cher au plus cher : la première ligne."
    )
