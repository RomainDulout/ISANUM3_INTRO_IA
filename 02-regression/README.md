# Séance 2 · Régression linéaire : comment une machine apprend

Au NB01, vous avez préparé les données. Vous entraînez ici un premier modèle : une droite qui estime le prix d'un diamant. On voit comment la machine trouve « la meilleure droite » (la descente de gradient, codée à la main) et comment mesurer si un modèle est bon.

## Pour commencer

Dans le Terminal, depuis le dossier du cours :
```bash
uv sync
uv run jupyter lab
```
Ouvrez ensuite `02-regression/NB02_regression.ipynb` et suivez les consignes. Utilisez **le même prénom qu'au NB01**.

## Au programme

| Exercice | Contenu | |
|---|---|---|
| 0 | Démarrer, recharger les données préparées | socle |
| 1 | Le modèle de référence et la MAE | socle |
| | Pause cours : ajustez la droite vous-même (curseurs) | socle |
| 2 | Régression simple : du carat au prix | socle |
| 3 | La descente de gradient, codée à la main | socle |
| 4 | Régression multiple : toutes les variables | socle |
| 5 | MAE, RMSE, R² : trois façons de mesurer l'erreur | socle |
| 6 | Entraînement contre test | socle |
| ⭐ | Les résidus et l'astuce du logarithme | bonus |
| ⭐⭐ | Prédire le prix des maisons d'Ames, sans guide | bonus |
| ⭐⭐⭐ | Divisez l'erreur par deux | bonus |

Après chaque exercice, une cellule `grader.check(...)` indique si la réponse est juste, ou donne un indice.

*Données : Diamonds (ggplot2, H. Wickham) ; Ames Housing (D. De Cock, Journal of Statistics Education, 2011).*

*© Romain Dulout, UPPA. Supports de cours originaux : toute reproduction ou réutilisation, même partielle, nécessite l'accord de l'auteur.*
