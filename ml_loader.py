"""
Chargement des pipelines supervisés, de K-Means, du background SHAP,
des importances et du dataset actualisé (628 utilisateurs).
Les pipelines contiennent leur propre prétraitement : aucune standardisation
ou aucun encodage manuel supplémentaire ne doit être appliqué dans Flask.
"""

from pathlib import Path
import json

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

FICHIERS_MODELES = {
    "Régression Logistique": "regression_logistique.pkl",
    "Arbre de Décision": "arbre_de_decision.pkl",
    "Random Forest": "random_forest.pkl",
    "SVM": "svm.pkl",
    "XGBoost": "xgboost.pkl",
}


def charger_tout():
    """Charge une seule fois tous les artefacts produits par train_models_628.py."""
    modeles = {}
    for nom, fichier in FICHIERS_MODELES.items():
        chemin = MODELS_DIR / fichier
        if not chemin.exists():
            raise FileNotFoundError(f"Modèle introuvable : {chemin}")
        modeles[nom] = joblib.load(chemin)

    fichiers_requis = [
        MODELS_DIR / "kmeans.pkl",
        MODELS_DIR / "scaler_cluster.pkl",
        MODELS_DIR / "shap_background_logistique.pkl",
        MODELS_DIR / "metadonnees.json",
        MODELS_DIR / "importance_permutation_logistique.csv",
        MODELS_DIR / "synthese_performances.csv",
        DATA_DIR / "dataset_complet.csv",
    ]
    absents = [str(p) for p in fichiers_requis if not p.exists()]
    if absents:
        raise FileNotFoundError(
            "Artefacts de la mise à jour 628 manquants :\n- " + "\n- ".join(absents)
        )

    kmeans = joblib.load(MODELS_DIR / "kmeans.pkl")
    scaler_cluster = joblib.load(MODELS_DIR / "scaler_cluster.pkl")
    shap_background = joblib.load(MODELS_DIR / "shap_background_logistique.pkl")

    with open(MODELS_DIR / "metadonnees.json", "r", encoding="utf-8") as f:
        metadonnees = json.load(f)

    dataset = pd.read_csv(DATA_DIR / "dataset_complet.csv")
    importance_permutation = pd.read_csv(
        MODELS_DIR / "importance_permutation_logistique.csv"
    )
    synthese_performances = pd.read_csv(
        MODELS_DIR / "synthese_performances.csv"
    )

    if len(dataset) != 628:
        raise ValueError(
            f"Le dataset chargé doit contenir 628 utilisateurs ; trouvé : {len(dataset)}."
        )
    if "fidelite" not in dataset.columns or "cluster" not in dataset.columns:
        raise ValueError("Le dataset doit contenir les colonnes fidelite et cluster.")

    print("✓ Cinq pipelines, K-Means, background SHAP, métriques et dataset 628 chargés.")

    return {
        "modeles": modeles,
        "kmeans": kmeans,
        "scaler_cluster": scaler_cluster,
        "shap_background": shap_background,
        "importance_permutation": importance_permutation,
        "synthese_performances": synthese_performances,
        "metadonnees": metadonnees,
        "dataset": dataset,
    }
