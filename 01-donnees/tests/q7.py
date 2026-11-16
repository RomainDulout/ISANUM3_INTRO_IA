import hashlib

from otter.test_files import test_case

OK_FORMAT = False
name = "q7"

EXPECTED_HASH = 'df7e70e5021544f4834bbee64a9e3789febc4be81470df629cad6ddb03320a5c'


@test_case(points=1, success_message="✅ Juste : le test a « fuité » dans la préparation des données.")
def test_leak(answer_leak):
    answer = str(answer_leak).strip().upper()
    assert answer in {"A", "B", "C", "D"}, "❌ Répondez par une lettre entre guillemets : \"A\", \"B\", \"C\" ou \"D\"."
    assert answer != "A", "❌ Les deux codes ne font pas la même chose : comparez sur quelles données chaque scaler fait son fit."
    assert answer != "C", "❌ Normaliser y n'est pas nécessaire ici : le problème concerne X et le moment du fit."
    assert answer != "D", "❌ Le random_state fixe seulement le tirage du découpage : le modèle ne mémorise rien d'une exécution à l'autre."
    assert hashlib.sha256(answer.encode()).hexdigest() == EXPECTED_HASH, "❌ Relisez les propositions."
