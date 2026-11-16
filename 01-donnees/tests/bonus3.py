import hashlib

from otter.test_files import test_case

OK_FORMAT = False
name = "bonus3"

# Empreinte de la bonne réponse (pour ne pas la donner en clair aux curieux)
EXPECTED_HASHES = {'6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b', '4e07408562bedb8b60ce05c1decfe3ad16b72230967de01f640b7e4729b49fce', 'ef2d127de37b942baad06145e54b0c619a1f22327b2ebbcfbec78f5564afe39d', '7902699be42c8a8e46fbbb4501726517e86b22c56a189f7625a6da49081b2451', '19581e27de7ced00ff1ce50b2047e7a567c76b1cbaebabe5ef03f7c3017bb5b7'}


@test_case(points=0, success_message="✅ Bonus 3 réussi : les 5 problèmes cachés sont trouvés.")
def test_bonus3(problems_found):
    found = set(problems_found)
    assert len(found) == 5, "❌ Il faut exactement 5 numéros (entre 1 et 10), sans doublon."
    good_ones = sum(hashlib.sha256(str(int(t)).encode()).hexdigest() in EXPECTED_HASHES for t in found)
    assert good_ones == 5, f"❌ {good_ones} problème(s) sur 5 correctement identifié(s). Continuez l'enquête."
