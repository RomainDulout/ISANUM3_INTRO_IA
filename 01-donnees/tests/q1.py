from otter.test_files import test_case

import course_tools as ct

OK_FORMAT = False
name = "q1"


@test_case(points=1, success_message="✅ Nombre de lignes correct.")
def test_rows(n_rows):
    expected = ct.load_dirty_diamonds().shape[0]
    assert n_rows == expected, "❌ n_rows : regardez df.shape, le premier nombre est le nombre de lignes."


@test_case(points=1, success_message="✅ Nombre de colonnes correct.")
def test_columns(n_columns):
    expected = ct.load_dirty_diamonds().shape[1]
    assert n_columns == expected, "❌ n_columns : regardez df.shape, le second nombre est le nombre de colonnes."


@test_case(points=1, success_message="✅ Nombre de colonnes numériques correct.")
def test_numeric(n_numeric):
    expected = ct.load_dirty_diamonds().select_dtypes("number").shape[1]
    assert n_numeric == expected, (
        "❌ n_numeric : comptez les colonnes de type float64 ou int64 dans df.info(), "
        "ou utilisez df.select_dtypes(\"number\").shape[1]."
    )
