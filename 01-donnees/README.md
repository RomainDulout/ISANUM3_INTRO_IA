# Séance 1 · Explorer et préparer des données

Un modèle d'IA n'est jamais meilleur que ses données. Dans cette séance, vous préparez un vrai jeu de données (le prix de ~54 000 diamants) comme le ferait un ou une data scientist : diagnostic, exploration, nettoyage, encodage, découpage.

## Pour commencer

Dans le Terminal, depuis le dossier du cours :
```bash
uv sync
uv run jupyter lab
```
Ouvrez ensuite `01-donnees/NB01_donnees.ipynb` et suivez les consignes.

## Au programme

| Exercice | Contenu | |
|---|---|---|
| 0 | Démarrer (votre prénom = votre échantillon personnel) | socle |
| 1 | Premier contact : taille, types, statistiques | socle |
| 2 | Diagnostic : valeurs manquantes, doublons, fautes de saisie | socle |
| 3 | Explorer avec des graphiques | socle |
| 4 | Nettoyer | socle |
| 5 | Encoder les variables textuelles | socle |
| 6 | Découper (train / test) et normaliser | socle |
| 7 | Cellule piégée : où est la fuite de données ? | socle |
| 8 | Sauvegarder pour la séance suivante | socle |
| ⭐ | Enquête libre | bonus |
| ⭐⭐ | Tout refaire en un `Pipeline` | bonus |
| ⭐⭐⭐ | Le fichier saboté | bonus |

Après chaque exercice, une cellule `grader.check(...)` indique si la réponse est juste, ou donne un indice.

*Données : Diamonds (ggplot2, H. Wickham), avec des défauts ajoutés pour l'exercice.*

*© Romain Dulout, UPPA. Supports de cours originaux : toute reproduction ou réutilisation, même partielle, nécessite l'accord de l'auteur.*
