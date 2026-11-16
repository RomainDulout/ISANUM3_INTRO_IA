import pandas as pd
from otter.test_files import test_case

import course_tools as ct

OK_FORMAT = False
name = "q8"


@test_case(points=1, success_message="✅ Fichier sauvegardé : il resservira pour la régression.")
def test_save():
    path = ct.personal_folder() / "diamonds_prepare.csv"
    assert path.exists(), "❌ Le fichier n'existe pas encore : exécutez la cellule de sauvegarde."
    df = pd.read_csv(path)
    assert "price" in df.columns and df.isnull().sum().sum() == 0, "❌ Le fichier sauvegardé n'est pas df_enc : vérifiez la variable utilisée."
