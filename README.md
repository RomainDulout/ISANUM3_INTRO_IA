# R5-07 · Introduction à l'intelligence artificielle

Supports du module **R5-07 Introduction à l'IA** (ISA NUM 3A, UPPA). Chaque séance a son dossier (`01-donnees`, puis les suivantes au fil du module), avec un notebook Jupyter et ses vérifications automatiques.

## Installation (une seule fois)

1. Installez **uv** dans le Terminal :
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   puis fermez et rouvrez le Terminal.
2. Récupérez ce dépôt (bouton vert **Code**, puis **Download ZIP**, ou `git clone`), et placez-vous dans son dossier.
3. Installez Python et les bibliothèques du cours :
   ```bash
   uv sync
   ```
   La première fois, cela prend 1 à 2 minutes.

## À chaque séance

```bash
uv run jupyter lab
```

Jupyter s'ouvre dans le navigateur. Ouvrez le notebook de la séance (par exemple `01-donnees/NB01_donnees.ipynb`) et suivez les consignes. Pour arrêter Jupyter : Ctrl+C dans le Terminal.

## Contenu

```
pyproject.toml, uv.lock   les bibliothèques du cours, aux mêmes versions pour tout le monde
course_tools/             petit module du cours : votre prénom, le chargement des données, les curseurs
donnees/                  les jeux de données (utilisables sans internet)
01-donnees/               séance 1 : explorer et préparer des données
```

Vos fichiers personnels (par exemple le tableau sauvegardé à la fin du NB01) vont dans `donnees_perso/`, créé automatiquement.

## Données

- Diamonds : ggplot2, H. Wickham. Les fichiers `diamonds_sali.csv` et `diamonds_sabote.csv` en sont dérivés, avec des défauts ajoutés pour les exercices.

## Auteur et droits

© Romain Dulout, UPPA. Les notebooks, les vérifications, le module `course_tools` et les énoncés sont des supports de cours originaux. Ils sont mis à disposition des étudiants du module pour leur usage personnel. Toute reproduction, diffusion ou réutilisation, même partielle, nécessite l'accord de l'auteur. Les jeux de données restent soumis à la licence de leur source.
