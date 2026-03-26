import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import gc

def model(features, test_features, encoding = 'ohe', n_folds = 5):
    """
    Entraîne et teste un modèle LightGBM en utilisant la validation croisée.
    Cette fonction est optimisée pour le suivi MLflow (via autolog dans le notebook).
    """
    
    # Extraction des identifiants et des étiquettes
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']
    labels = features['TARGET']
    
    # Suppression des colonnes non prédictives
    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns = ['SK_ID_CURR'])
    
    # Gestion de l'encodage
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)
        features, test_features = features.align(test_features, join = 'inner', axis = 1)
        cat_indices = 'auto'
    elif encoding == 'le':
        label_encoder = LabelEncoder()
        cat_indices = []
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))
                cat_indices.append(i)
    else:
        raise ValueError("L'encodage doit être 'ohe' ou 'le'")
        
    print('Training Data Shape: ', features.shape)
    feature_names = list(features.columns)
    
    # Préparation de la validation croisée
    k_fold = KFold(n_splits = n_folds, shuffle = True, random_state = 50)
    feature_importance_values = np.zeros(len(feature_names))
    test_predictions = np.zeros(test_features.shape[0])
    out_of_fold = np.zeros(features.shape[0])
    
    valid_scores = []
    train_scores = []
    
    for train_indices, valid_indices in k_fold.split(features):
        # Utilisation de .values pour s'assurer d'avoir des arrays numpy
        train_features, train_labels = features.values[train_indices], labels[train_indices]
        valid_features, valid_labels = features.values[valid_indices], labels[valid_indices]
        
        # Création du modèle (Configuré pour le déséquilibre des classes)
        clf = lgb.LGBMClassifier(n_estimators=10000, 
                                 objective = 'binary', 
                                 class_weight = 'balanced', 
                                 learning_rate = 0.05, 
                                 reg_alpha = 0.1, 
                                 reg_lambda = 0.1, 
                                 subsample = 0.8, 
                                 n_jobs = -1, 
                                 random_state = 50)
        
        # Callbacks pour l'early stopping et les logs (Syntaxe 2024)
        callbacks = [
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=200)
        ]
        
        # Entraînement
        clf.fit(train_features, train_labels, 
                eval_metric = 'auc',
                eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                eval_names = ['valid', 'train'], 
                categorical_feature = cat_indices,
                callbacks = callbacks)
        
        # Enregistrement
        best_iteration = clf.best_iteration_
        feature_importance_values += clf.feature_importances_ / k_fold.n_splits
        test_predictions += clf.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits
        out_of_fold[valid_indices] = clf.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
        
        valid_scores.append(clf.best_score_['valid']['auc'])
        train_scores.append(clf.best_score_['train']['auc'])
        
        gc.collect()
        
    # Résultats finaux
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    
    valid_auc = roc_auc_score(labels, out_of_fold)
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))
    
    metrics = pd.DataFrame({
        'fold': list(range(n_folds)) + ['overall'],
        'train': train_scores,
        'valid': valid_scores
    }) 
    
    return submission, feature_importances, metrics