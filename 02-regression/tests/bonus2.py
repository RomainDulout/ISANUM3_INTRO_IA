from otter.test_files import test_case

OK_FORMAT = False
name = "bonus2"


@test_case(points=1, success_message="✅ Votre régression bat le modèle de référence sur un jeu de données que vous avez préparé seul ou seule.")
def test_ames(mae_ames_baseline, mae_ames):
    assert mae_ames_baseline is not ... and mae_ames is not ..., "❌ Remplacez les ... de mae_ames_baseline et mae_ames."
    assert 30_000 < float(mae_ames_baseline) < 90_000, (
        "❌ mae_ames_baseline semble fausse : le modèle de référence prédit la moyenne de SalePrice (entraînement) pour toutes les maisons du test."
    )
    assert 0 < float(mae_ames) < float(mae_ames_baseline), "❌ Votre régression ne fait pas mieux que le modèle de référence."
