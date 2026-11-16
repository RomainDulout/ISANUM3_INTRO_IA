from otter.test_files import test_case

import course_tools as ct

OK_FORMAT = False
name = "q2"


@test_case(points=1, success_message="✅ Total des valeurs manquantes correct.")
def test_missing(n_missing):
    df = ct.load_dirty_diamonds()
    per_column = df.isnull().sum()
    expected = per_column.sum()
    assert n_missing != per_column.max(), (
        "❌ n_missing : vous avez pris une seule colonne. Il faut le total de toutes les colonnes : df.isnull().sum().sum()."
    )
    assert n_missing == expected, "❌ n_missing : additionnez les valeurs manquantes de toutes les colonnes."


@test_case(points=1, success_message="✅ Nombre de doublons correct.")
def test_duplicates(n_duplicates):
    expected = ct.load_dirty_diamonds().duplicated().sum()
    assert n_duplicates == expected, "❌ n_duplicates : utilisez df.duplicated().sum()."


@test_case(points=1, success_message="✅ Nombre de modalités de cut correct : il y en a plus que 5, à cause des fautes de saisie.")
def test_categories(n_cut_categories):
    expected = ct.load_dirty_diamonds()["cut"].nunique()
    assert n_cut_categories != 5, (
        "❌ n_cut_categories : il devrait y en avoir 5 en théorie, mais regardez bien df[\"cut\"].value_counts() : "
        "certaines valeurs sont mal écrites."
    )
    assert n_cut_categories == expected, "❌ n_cut_categories : utilisez df[\"cut\"].nunique()."
