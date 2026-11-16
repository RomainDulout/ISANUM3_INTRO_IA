from otter.test_files import test_case

OK_FORMAT = False
name = "q5"


@test_case(points=1, success_message="✅ Bonne prédiction calculée : 7 + 5 + 7 + 8 = 27 colonnes en one-hot.")
def test_onehot(n_columns_onehot):
    assert n_columns_onehot == 27, "❌ Exécutez la cellule qui calcule n_columns_onehot sans la modifier."


@test_case(points=1, success_message="✅ Ordre des couleurs correct.")
def test_color(color_order):
    assert list(color_order) == ["J", "I", "H", "G", "F", "E", "D"], (
        "❌ color_order : il faut les 7 lettres, de la moins bonne (J) à la meilleure (D), chacune entre guillemets."
    )


@test_case(points=1, success_message="✅ Ordre des puretés correct.")
def test_clarity(clarity_order):
    assert list(clarity_order) == ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], (
        "❌ clarity_order : il faut les 8 valeurs dans l'ordre du tableau de l'exercice 0, de I1 à IF."
    )


@test_case(points=1, success_message="✅ df_enc ne contient plus que des nombres.")
def test_df_enc(df_enc):
    for column in ["cut", "color", "clarity"]:
        assert df_enc[column].isnull().sum() == 0, (
            f"❌ La colonne {column} contient des valeurs vides après encodage : une catégorie manque dans votre liste "
            "ou est mal écrite."
        )
    assert df_enc.select_dtypes("number").shape[1] == df_enc.shape[1], "❌ df_enc contient encore des colonnes de texte."
