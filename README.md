# Projet 6 - Implémentation d'un modèle de scoring

## Description
Ce projet consiste à développer un algorithme de classification pour prédire le risque de défaut de paiement d'un client pour la société "Home Credit".

## Structure du projet
- `notebooks/` : Contient l'analyse exploratoire et la modélisation.
- `data/` : (Non inclus sur Git) Contient les fichiers CSV de Kaggle.
- `mlruns/` : Dossier de suivi des expériences MLflow.

## Installation
1. Cloner le dépôt.
2. Créer un environnement virtuel : `python3 -m venv .venv`.
3. Installer les dépendances : `pip install -r requirements.txt`.
4. Télécharger les données sur [Kaggle Home Credit](https://www.kaggle.com/c/home-credit-default-risk/data) et les placer dans le dossier `/data`.

## Résultats (Baseline)
- **Modèle** : LightGBM
- **Métrique AUC** : 0.766