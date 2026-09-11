"""
Chargement des modèles ML, du scaler, de K-Means, des encodeurs et
du dataset. Chargé UNE SEULE FOIS au démarrage de l'application
(pas à chaque requête), pour des performances optimales.
"""

import os
import json
import joblib
import pandas as pd

RACINE = os.path.dirname(os.path.abspath(__file__))
DOSSIER_MODELES = os.path.join(RACINE, 'models')
DOSSIER_DATA = os.path.join(RACINE, 'data')


def _charger_json(nom_fichier):
    chemin = os.path.join(DOSSIER_MODELES, nom_fichier)
    with open(chemin, 'r', encoding='utf-8') as f:
        return json.load(f)


def charger_tout():
    """Charge tous les artefacts nécessaires et les retourne dans un
    dictionnaire unique. À appeler une seule fois au démarrage de
    l'app (voir app.py)."""

    modeles = {
        'Régression Logistique': joblib.load(os.path.join(DOSSIER_MODELES, 'regression_logistique.pkl')),
        'Arbre de Décision': joblib.load(os.path.join(DOSSIER_MODELES, 'arbre_de_decision.pkl')),
        'Random Forest': joblib.load(os.path.join(DOSSIER_MODELES, 'random_forest.pkl')),
        'SVM': joblib.load(os.path.join(DOSSIER_MODELES, 'svm.pkl')),
        'XGBoost': joblib.load(os.path.join(DOSSIER_MODELES, 'xgboost.pkl')),
    }

    scaler = joblib.load(os.path.join(DOSSIER_MODELES, 'scaler.pkl'))
    kmeans = joblib.load(os.path.join(DOSSIER_MODELES, 'kmeans.pkl'))
    scaler_cluster = joblib.load(os.path.join(DOSSIER_MODELES, 'scaler_cluster.pkl'))
    encodeurs = joblib.load(os.path.join(DOSSIER_MODELES, 'encodeurs_complet.pkl'))
    metadonnees = _charger_json('metadonnees.json')

    dataset = pd.read_csv(os.path.join(DOSSIER_DATA, 'dataset_complet.csv'))

    print("✓ Modèles, scaler, K-Means, encodeurs et dataset chargés avec succès.")

    return {
        'modeles': modeles,
        'scaler': scaler,
        'kmeans': kmeans,
        'scaler_cluster': scaler_cluster,
        'encodeurs': encodeurs,
        'metadonnees': metadonnees,
        'dataset': dataset,
    }