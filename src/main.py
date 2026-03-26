import os
import pandas as pd
import mlflow
from src.data_cleaning import (
    handle_anomalies, 
    feature_engineering_domain, 
    encode_categorical
)

# 1. Configuration de l'expérience MLflow
mlflow.set_experiment("Credit_Scoring_Data_Prep")

def run_data_pipeline():
    # Démarre un run MLflow pour tracer cette préparation de données
    with mlflow.start_run(run_name="Initial_Cleaning_and_Feature_Engineering"):
        
        print("--- Chargement des données brutes ---")
        # Chemins relatifs vers vos fichiers Kaggle
        train_path = 'data/raw/application_train.csv'
        test_path = 'data/raw/application_test.csv'
        
        app_train = pd.read_csv(train_path)
        app_test = pd.read_csv(test_path)
        
        # Log des paramètres de base dans MLflow
        mlflow.log_param("raw_train_shape", app_train.shape)
        mlflow.log_param("raw_test_shape", app_test.shape)

        # --- ETAPE A : Traitement des anomalies ---
        # On utilise la logique du notebook : correction de DAYS_EMPLOYED
        print("Nettoyage des anomalies...")
        app_train = handle_anomalies(app_train)
        app_test = handle_anomalies(app_test)
        mlflow.log_param("days_employed_anomaly_fixed", True)

        # --- ETAPE B : Feature Engineering métier ---
        # Création des ratios CREDIT_INCOME, ANNUITY_INCOME, etc.
        print("Création des variables métier...")
        app_train = feature_engineering_domain(app_train)
        app_test = feature_engineering_domain(app_test)
        mlflow.log_param("domain_features_created", ["CREDIT_INCOME", "ANNUITY_INCOME", "TERM", "EMPLOYED_PERCENT"])

        # --- ETAPE C : Encodage et Alignement ---
        # LabelEncoding (2 classes) et One-Hot (>2 classes)
        print("Encodage des variables catégorielles...")
        app_train, app_test = encode_categorical(app_train, app_test)
        mlflow.log_param("final_feature_count", app_train.shape[1])

        # --- ETAPE D : Sauvegarde ---
        print("Sauvegarde des datasets préparés...")
        os.makedirs('data/processed', exist_ok=True)
        app_train.to_csv('data/processed/train_cleaned.csv', index=False)
        app_test.to_csv('data/processed/test_cleaned.csv', index=False)
        
        # On log le fichier final comme un artefact pour le retrouver facilement
        mlflow.log_artifact('data/processed/train_cleaned.csv')
        
        print(f"Etape 1 terminée. Dataset prêt : {app_train.shape}")

if __name__ == "__main__":
    run_data_pipeline()