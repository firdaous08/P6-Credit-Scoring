import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def handle_anomalies(df):
    """
    Traite les anomalies identifiées dans le notebook Kaggle.
    Notamment la valeur 365243 dans DAYS_EMPLOYED.
    """
    # Création d'une colonne flag pour l'anomalie
    df['DAYS_EMPLOYED_ANOM'] = df["DAYS_EMPLOYED"] == 365243
    
    # Remplacement par NaN pour permettre l'imputation ultérieure
    df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    
    # Conversion de DAYS_BIRTH en valeurs positives pour faciliter l'interprétation
    df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH'])
    
    return df

def feature_engineering_domain(df):
    """
    Crée des variables métier suggérées dans le kernel d'introduction.
    """
    # Ratio du montant du crédit par rapport au revenu du client
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    
    # Ratio de l'annuité par rapport au revenu du client
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    
    # Durée théorique du prêt (Annuité / Crédit)
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    
    # Pourcentage des jours travaillés par rapport à l'âge
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    
    return df

def encode_categorical(train, test):
    """
    Applique le Label Encoding (2 classes) et le One-Hot Encoding (>2 classes) 
    comme décrit dans le notebook.
    """
    le = LabelEncoder()
    
    for col in train.columns:
        if train[col].dtype == 'object':
            # Label Encoding si 2 catégories ou moins
            if len(list(train[col].unique())) <= 2:
                le.fit(train[col])
                train[col] = le.transform(train[col])
                test[col] = le.transform(test[col])
    
    # One-Hot Encoding pour le reste
    train = pd.get_dummies(train)
    test = pd.get_dummies(test)
    
    # Alignement des colonnes entre train et test
    train_labels = train['TARGET']
    train, test = train.align(test, join='inner', axis=1)
    train['TARGET'] = train_labels
    
    return train, test

def business_cost_score(y_true, y_pred_proba, threshold=0.5):
    """
    Calcule le coût métier personnalisé demandé par Michaël.
    Coût = 10 * Faux Négatifs + 1 * Faux Positifs.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Identification des FN (Mauvais clients prédits bons) et FP (Bons clients prédits mauvais)
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    
    # Application du ratio de coût 10:1
    cost = (10 * fn) + (1 * fp)
    return cost