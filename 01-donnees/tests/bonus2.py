import numpy as np
from otter.test_files import test_case

OK_FORMAT = False
name = "bonus2"


@test_case(points=0, success_message="✅ La chaîne de préparation fonctionne.")
def test_pipeline(preprocessing, X_raw_train, X_raw_test, Xp_train, Xp_test):
    assert hasattr(preprocessing, "transform"), "❌ preprocessing doit être un ColumnTransformer (ou un Pipeline)."
    assert Xp_train.shape[0] == X_raw_train.shape[0] and Xp_test.shape[0] == X_raw_test.shape[0], (
        "❌ Xp_train et Xp_test doivent avoir autant de lignes que X_raw_train et X_raw_test."
    )
    assert Xp_train.shape[1] == 9, "❌ On attend 9 colonnes : 6 numériques et 3 encodées."
    assert not np.isnan(np.asarray(Xp_train, dtype=float)).any(), "❌ Il reste des valeurs manquantes : avez-vous mis un SimpleImputer ?"
