# -*- coding: utf-8 -*-
"""
Mise à jour analytique de la plateforme Orange Money - 628 utilisateurs.

Ce script :
1) évalue les 5 modèles par validation croisée imbriquée répétée
   (5 plis externes x 10 répétitions ; GridSearchCV interne à 3 plis) ;
2) calcule l'importance par permutation de la régression logistique
   sur les 50 plis externes ;
3) réentraîne chacun des 5 meilleurs pipelines sur les 628 observations
   avec un GridSearchCV final et sauvegarde les pipelines ;
4) réentraîne K-Means (K=5) sur les 628 observations ;
5) met à jour le dataset et les métadonnées de la plateforme.

À exécuter depuis le dossier racine Plateforme_flask.
Le script ne modifie ni app.py, ni ml_loader.py, ni les templates/CSS.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURATION GÉNÉRALE
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

FICHIER_SOURCE = DATA_DIR / "dataset_complet.csv"
FICHIER_DATASET_PLATEFORME = DATA_DIR / "dataset_complet.csv"

RANDOM_STATE = 42
N_SPLITS_EXTERNES = 5
N_REPEATS_EXTERNES = 10
N_SPLITS_INTERNES = 3
N_REPEATS_PERMUTATION = 10

# Les 21 variables source exactement utilisées dans le modèle principal.
COLONNES_MODELE = [
    "tranche_age",
    "sexe",
    "situation_professionnelle",
    "anciennete_usage",
    "frequence_usage",
    "sat_globale",
    "perception_frais",
    "presence_probleme",
    "operateur_prefere",
    "motif_principal",
    "pb_aucun",
    "pb_autre",
    "pb_lenteur",
    "pb_probleme_de_reseau",
    "pb_service_client",
    "pb_transaction_echouee",
    "serv_achat_de_credit",
    "serv_paiement_de_factures",
    "serv_paiement_marchand",
    "serv_reception_dargent",
    "serv_transfert_dargent",
]

COLONNES_NUM = [
    "tranche_age",
    "anciennete_usage",
    "frequence_usage",
    "sat_globale",
    "perception_frais",
]

COLONNES_CAT = [
    "sexe",
    "situation_professionnelle",
    "operateur_prefere",
    "motif_principal",
]

COLONNES_CLUSTER = [
    "sat_globale",
    "frequence_usage",
    "anciennete_usage",
    "perception_frais",
]

# Correspondance des clusters obtenue dans l'analyse K-Means sur 628 personnes.
NOMS_CLUSTERS = {
    "0": "Utilisateurs à Risque",
    "1": "Fidèles Engagés",
    "2": "Nouveaux Utilisateurs",
    "3": "Utilisateurs Captifs Insatisfaits",
    "4": "Fidèles Passifs",
}

# Nom des fichiers conservés pour rester compatibles avec l'organisation actuelle.
NOMS_FICHIERS_MODELES = {
    "Régression Logistique": "regression_logistique.pkl",
    "Arbre de Décision": "arbre_de_decision.pkl",
    "Random Forest": "random_forest.pkl",
    "SVM": "svm.pkl",
    "XGBoost": "xgboost.pkl",
}


# ============================================================
# 2. CHARGEMENT ET CONTRÔLES DU DATASET
# ============================================================
def charger_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if not FICHIER_SOURCE.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {FICHIER_SOURCE}\n"
            "Vérifie que data/dataset_complet.csv contient la nouvelle base de 628 utilisateurs."
        )

    df = pd.read_csv(FICHIER_SOURCE)

    if len(df) != 628:
        raise ValueError(
            f"Le dataset doit contenir exactement 628 lignes ; trouvé : {len(df)}."
        )

    colonnes_absentes = [c for c in COLONNES_MODELE + ["fidelite"] if c not in df.columns]
    if colonnes_absentes:
        raise ValueError(f"Colonnes manquantes dans le dataset : {colonnes_absentes}")

    # La cible doit être strictement binaire et conforme au modèle principal.
    df["fidelite"] = pd.to_numeric(df["fidelite"], errors="raise").astype(int)
    distribution = df["fidelite"].value_counts().to_dict()
    if distribution != {1: 573, 0: 55}:
        raise ValueError(
            "Distribution inattendue de la cible. Attendu : 573 fidèles (1) et "
            f"55 non-fidèles (0). Obtenu : {distribution}"
        )

    X = df[COLONNES_MODELE].copy()
    y = df["fidelite"].copy()

    if X.isna().any().any():
        manquantes = X.isna().sum()
        manquantes = manquantes[manquantes > 0].to_dict()
        raise ValueError(f"Le dataset comporte encore des valeurs manquantes : {manquantes}")

    # Contrôle des catégories brutes : elles doivent être conservées sous forme de texte.
    for col in COLONNES_CAT:
        X[col] = X[col].astype(str).str.strip()
        if (X[col] == "").any():
            raise ValueError(f"La variable catégorielle {col} contient une chaîne vide.")

    # Les indicateurs binaires et ordinals doivent être numériques.
    for col in set(COLONNES_MODELE) - set(COLONNES_CAT):
        X[col] = pd.to_numeric(X[col], errors="raise")

    colonnes_binaires = [c for c in COLONNES_MODELE if c not in COLONNES_NUM + COLONNES_CAT]
    for col in colonnes_binaires:
        valeurs = set(X[col].dropna().unique().tolist())
        if not valeurs.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(f"La variable binaire {col} contient des valeurs inattendues : {valeurs}")
        X[col] = X[col].astype(int)

    print("\n--- CONTRÔLE DU DATASET ---")
    print(f"Dimensions X : {X.shape}")
    print(f"Dimensions y : {y.shape}")
    print(f"Distribution cible : {distribution}")
    print(f"Valeurs manquantes dans X : {int(X.isna().sum().sum())}")
    print("Variables catégorielles :", COLONNES_CAT)
    print("Variables numériques :", COLONNES_NUM)
    print("Variables binaires en passthrough :", colonnes_binaires)

    return df, X, y


# ============================================================
# 3. PRÉTRAITEMENT ET MODÈLES
# ============================================================
def construire_preprocesseur() -> ColumnTransformer:
    # Même logique que le mémoire : StandardScaler, OneHotEncoder,
    # puis passage direct des 12 variables binaires.
    return ColumnTransformer(
        transformers=[
            ("numeriques", StandardScaler(), COLONNES_NUM),
            ("nominales", OneHotEncoder(handle_unknown="ignore"), COLONNES_CAT),
        ],
        remainder="passthrough",
    )


def construire_modeles_et_grilles():
    modeles = {
        "Régression Logistique": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {"clf__C": [0.1, 1, 10]},
        ),
        "Arbre de Décision": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "clf__max_depth": [3, 5, None],
                "clf__min_samples_leaf": [1, 5],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 10],
            },
        ),
        "SVM": (
            SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
            {
                "clf__C": [0.1, 1, 10],
                "clf__gamma": ["scale"],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=1.0,
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=0,
            ),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [3, 5],
            },
        ),
    }
    return modeles


def construire_recherche(nom_modele, estimateur, grille, cv_interne, scoring):
    pipeline = Pipeline(
        steps=[
            ("preprocesseur", construire_preprocesseur()),
            ("clf", estimateur),
        ]
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid=grille,
        scoring=scoring,
        cv=cv_interne,
        refit=True,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    )


# ============================================================
# 4. MÉTRIQUES DE LA CLASSE 0
# ============================================================
def calculer_metriques(y_true, y_pred, proba_classe0):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_classe0": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_classe0": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_classe0": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_classe1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc_classe0": roc_auc_score((np.asarray(y_true) == 0).astype(int), proba_classe0),
        "average_precision_classe0": average_precision_score(
            (np.asarray(y_true) == 0).astype(int), proba_classe0
        ),
    }


def obtenir_proba_classe0(recherche, X_test):
    probabilites = recherche.predict_proba(X_test)
    classes = recherche.best_estimator_.named_steps["clf"].classes_
    position_0 = int(np.where(classes == 0)[0][0])
    return probabilites[:, position_0]


# ============================================================
# 5. VALIDATION CROISÉE IMBRIQUÉE DES 5 MODÈLES
# ============================================================
def evaluer_modeles(X, y):
    scoring = make_scorer(f1_score, pos_label=0, zero_division=0)
    cv_externe = RepeatedStratifiedKFold(
        n_splits=N_SPLITS_EXTERNES,
        n_repeats=N_REPEATS_EXTERNES,
        random_state=RANDOM_STATE,
    )
    cv_interne = StratifiedKFold(
        n_splits=N_SPLITS_INTERNES,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Les mêmes 50 découpages sont utilisés pour tous les modèles.
    plis_externes = list(cv_externe.split(X, y))
    modeles_grilles = construire_modeles_et_grilles()

    lignes_resultats = []
    lignes_importance = []
    total = len(modeles_grilles) * len(plis_externes)
    compteur = 0

    print("\n--- VALIDATION CROISÉE IMBRIQUÉE ---")
    print(f"{len(plis_externes)} plis externes par modèle ; {total} évaluations au total.")

    for nom_modele, (estimateur, grille) in modeles_grilles.items():
        print(f"\nModèle : {nom_modele}")

        for numero_pli, (idx_train, idx_test) in enumerate(plis_externes, start=1):
            compteur += 1
            X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
            y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]

            recherche = construire_recherche(
                nom_modele, estimateur, grille, cv_interne, scoring
            )
            recherche.fit(X_train, y_train)

            y_pred = recherche.predict(X_test)
            proba0 = obtenir_proba_classe0(recherche, X_test)
            metriques = calculer_metriques(y_test, y_pred, proba0)

            lignes_resultats.append({
                "modele": nom_modele,
                "pli_externe": numero_pli,
                **metriques,
                "meilleurs_parametres": json.dumps(recherche.best_params_, ensure_ascii=False),
                "f1_interne_classe0": float(recherche.best_score_),
            })

            # Importance par permutation uniquement pour la régression logistique.
            # Le modèle évalué est celui ajusté sur le pli d'entraînement ;
            # la permutation est réalisée sur le pli externe de test.
            if nom_modele == "Régression Logistique":
                importance = permutation_importance(
                    recherche.best_estimator_,
                    X_test,
                    y_test,
                    scoring=scoring,
                    n_repeats=N_REPEATS_PERMUTATION,
                    random_state=RANDOM_STATE + numero_pli,
                    n_jobs=1,
                )
                for i, variable in enumerate(COLONNES_MODELE):
                    lignes_importance.append({
                        "pli": numero_pli,
                        "variable": variable,
                        "importance_moyenne": float(importance.importances_mean[i]),
                        "ecart_type_repetitions": float(importance.importances_std[i]),
                    })

            if numero_pli % 5 == 0:
                print(f"  {numero_pli}/{len(plis_externes)} plis terminés")

    resultats = pd.DataFrame(lignes_resultats)
    resultats.to_csv(MODELS_DIR / "resultats_validation_croisee.csv", index=False, encoding="utf-8-sig")

    colonnes_metriques = [
        "accuracy", "balanced_accuracy", "precision_classe0", "recall_classe0",
        "f1_classe0", "f1_classe1", "roc_auc_classe0", "average_precision_classe0",
    ]
    synthese_lignes = []
    for nom_modele, groupe in resultats.groupby("modele", sort=False):
        ligne = {"modele": nom_modele, "nombre_plis": len(groupe)}
        for metrique in colonnes_metriques:
            ligne[f"{metrique}_moyenne"] = groupe[metrique].mean()
            ligne[f"{metrique}_ecart_type"] = groupe[metrique].std(ddof=1)
        synthese_lignes.append(ligne)

    synthese = pd.DataFrame(synthese_lignes)
    synthese.to_csv(MODELS_DIR / "synthese_performances.csv", index=False, encoding="utf-8-sig")

    # Synthèse des importances par permutation (21 variables source).
    details_importance = pd.DataFrame(lignes_importance)
    details_importance.to_csv(
        MODELS_DIR / "details_importance_permutation_logistique.csv",
        index=False,
        encoding="utf-8-sig",
    )
    synthese_importance = (
        details_importance.groupby("variable")
        .agg(
            importance_moyenne=("importance_moyenne", "mean"),
            ecart_type_entre_plis=("importance_moyenne", "std"),
            importance_min=("importance_moyenne", "min"),
            importance_max=("importance_moyenne", "max"),
            proportion_plis_positifs=("importance_moyenne", lambda s: (s > 0).mean()),
        )
        .reset_index()
    )
    synthese_importance["rang"] = synthese_importance["importance_moyenne"].rank(
        ascending=False, method="min"
    )
    synthese_importance = synthese_importance.sort_values(
        "importance_moyenne", ascending=False
    ).reset_index(drop=True)
    synthese_importance.to_csv(
        MODELS_DIR / "importance_permutation_logistique.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n--- SYNTHÈSE DES PERFORMANCES (moyenne et écart-type sur 50 plis) ---")
    print(synthese.round(4).to_string(index=False))
    print("\n--- IMPORTANCE PAR PERMUTATION : RÉGRESSION LOGISTIQUE ---")
    print(synthese_importance.round(4).to_string(index=False))

    return synthese, synthese_importance


# ============================================================
# 6. RÉENTRAÎNEMENT FINAL DES 5 PIPELINES SUR LES 628 UTILISATEURS
# ============================================================
def entrainer_modeles_finaux(X, y):
    scoring = make_scorer(f1_score, pos_label=0, zero_division=0)
    cv_interne = StratifiedKFold(
        n_splits=N_SPLITS_INTERNES,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    modeles_grilles = construire_modeles_et_grilles()
    parametres_finaux = {}
    pipelines_finaux = {}

    print("\n--- GRIDSEARCHCV FINAL SUR LES 628 UTILISATEURS ---")
    for nom_modele, (estimateur, grille) in modeles_grilles.items():
        print(f"Entraînement final : {nom_modele}")
        recherche = construire_recherche(
            nom_modele, estimateur, grille, cv_interne, scoring
        )
        recherche.fit(X, y)

        pipeline_final = recherche.best_estimator_
        nom_fichier = NOMS_FICHIERS_MODELES[nom_modele]
        joblib.dump(pipeline_final, MODELS_DIR / nom_fichier, compress=3)

        pipelines_finaux[nom_modele] = pipeline_final
        parametres_finaux[nom_modele] = {
            "fichier": nom_fichier,
            "meilleurs_parametres": recherche.best_params_,
            "meilleur_f1_cv_interne_classe0": float(recherche.best_score_),
        }
        print(f"  ✓ {nom_fichier} ; paramètres : {recherche.best_params_}")

    return pipelines_finaux, parametres_finaux


# ============================================================
# 7. SEGMENTATION K-MEANS SUR LES 628 UTILISATEURS
# ============================================================
def recalculer_segmentation(df):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X_cluster = df[COLONNES_CLUSTER].astype(float).copy()
    scaler_cluster = StandardScaler()
    X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_cluster_scaled)

    effectifs = pd.Series(clusters).value_counts().sort_index().to_dict()
    effectifs_attendus = {0: 151, 1: 155, 2: 64, 3: 89, 4: 169}
    if effectifs != effectifs_attendus:
        raise ValueError(
            "Les effectifs K-Means ne correspondent pas à la segmentation validée. "
            f"Attendu : {effectifs_attendus} ; obtenu : {effectifs}. "
            "Vérifie l'ordre des observations, les variables et leur codage."
        )

    df_final = df.copy()
    df_final["cluster"] = clusters.astype(int)

    joblib.dump(kmeans, MODELS_DIR / "kmeans.pkl", compress=3)
    joblib.dump(scaler_cluster, MODELS_DIR / "scaler_cluster.pkl", compress=3)
    df_final.to_csv(FICHIER_DATASET_PLATEFORME, index=False, encoding="utf-8-sig")

    score_silhouette = float(silhouette_score(X_cluster_scaled, clusters))

    print("\n--- SEGMENTATION K-MEANS ACTUALISÉE ---")
    print(f"Coefficient de silhouette : {score_silhouette:.4f}")
    print("Effectifs des clusters :", effectifs)
    print("✓ kmeans.pkl et scaler_cluster.pkl sauvegardés.")
    print(f"✓ Dataset de la plateforme sauvegardé : {FICHIER_DATASET_PLATEFORME}")

    return df_final, score_silhouette


# ============================================================
# 8. BACKGROUND SHAP ET MÉTADONNÉES
# ============================================================
def preparer_shap_et_metadonnees(df_final, X, pipelines_finaux, parametres_finaux, score_silhouette):
    pipeline_lr = pipelines_finaux["Régression Logistique"]
    preprocesseur = pipeline_lr.named_steps["preprocesseur"]

    X_transforme = preprocesseur.transform(X)
    try:
        from scipy import sparse
        if sparse.issparse(X_transforme):
            X_transforme = X_transforme.toarray()
    except ImportError:
        pass
    X_transforme = np.asarray(X_transforme, dtype=float)

    rng = np.random.default_rng(RANDOM_STATE)
    taille_background = min(100, len(X_transforme))
    indices_background = np.sort(
        rng.choice(len(X_transforme), size=taille_background, replace=False)
    )
    background = X_transforme[indices_background]
    joblib.dump(background, MODELS_DIR / "shap_background_logistique.pkl", compress=3)

    noms_variables_transforme = preprocesseur.get_feature_names_out().tolist()

    # Valeurs de référence (mode) pour les champs non demandés par le formulaire.
    valeurs_reference = {}
    for col in COLONNES_MODELE:
        mode = X[col].mode(dropna=True)
        if not mode.empty:
            valeur = mode.iloc[0]
            if isinstance(valeur, np.generic):
                valeur = valeur.item()
            valeurs_reference[col] = valeur

    # Modes des indicateurs de problèmes conditionnés à presence_probleme,
    # utiles au formulaire qui ne détaille pas la nature du problème.
    df_problemes = df_final[df_final["presence_probleme"] == 1]
    for col in [c for c in COLONNES_MODELE if c.startswith("pb_")]:
        if not df_problemes.empty:
            mode_pb = df_problemes[col].mode(dropna=True)
            if not mode_pb.empty:
                valeurs_reference[col] = int(mode_pb.iloc[0])
    valeurs_reference["pb_aucun"] = 0

    metadonnees = {
        "version_schema": "628_v1",
        "date_entrainement": datetime.now().isoformat(timespec="seconds"),
        "n_observations": int(len(df_final)),
        "distribution_cible": {"0": 55, "1": 573},
        "colonnes_modele": COLONNES_MODELE,
        "colonnes_num_scaler": COLONNES_NUM,
        "colonnes_categorielles": COLONNES_CAT,
        "colonnes_binaires_passthrough": [
            c for c in COLONNES_MODELE if c not in COLONNES_NUM + COLONNES_CAT
        ],
        "colonnes_clustering": COLONNES_CLUSTER,
        "noms_clusters": NOMS_CLUSTERS,
        "effectifs_clusters": {
            str(k): int(v) for k, v in pd.Series(df_final["cluster"]).value_counts().sort_index().items()
        },
        "k_optimal": 5,
        "silhouette_kmeans": score_silhouette,
        "valeurs_reference_prediction": valeurs_reference,
        "parametres_finaux": parametres_finaux,
        "variables_transforme_shap": noms_variables_transforme,
        "taille_background_shap": int(taille_background),
        "interpretabilite": {
            "modele": "Régression Logistique",
            "shap": "LinearExplainer",
            "importance_globale": "Permutation importance sur F1 de la classe 0",
        },
        "versions": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }

    with open(MODELS_DIR / "metadonnees.json", "w", encoding="utf-8") as f:
        json.dump(metadonnees, f, ensure_ascii=False, indent=2)

    with open(MODELS_DIR / "versions_entrainement.json", "w", encoding="utf-8") as f:
        json.dump(metadonnees["versions"], f, ensure_ascii=False, indent=2)

    print("✓ Background SHAP sauvegardé.")
    print("✓ Métadonnées 628 sauvegardées.")


# ============================================================
# 9. SAUVEGARDE DE SÉCURITÉ AVANT REMPLACEMENT DES ARTEFACTS
# ============================================================
def sauvegarder_anciens_artefacts():
    import shutil

    horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
    dossier_backup = BASE_DIR / "backups" / f"avant_maj_628_{horodatage}"
    dossier_backup.mkdir(parents=True, exist_ok=True)

    # Sauvegarde des modèles et métadonnées déjà présents.
    if MODELS_DIR.exists():
        dossier_models_backup = dossier_backup / "models"
        dossier_models_backup.mkdir(parents=True, exist_ok=True)
        for fichier in MODELS_DIR.iterdir():
            if fichier.is_file():
                shutil.copy2(fichier, dossier_models_backup / fichier.name)

    # Sauvegarde du dataset actuellement présent avant sa réécriture par K-Means.
    if FICHIER_DATASET_PLATEFORME.exists():
        shutil.copy2(FICHIER_DATASET_PLATEFORME, dossier_backup / "dataset_complet.csv")

    print(f"✓ Sauvegarde de sécurité créée : {dossier_backup}")


# ============================================================
# 10. EXÉCUTION
# ============================================================
def main():
    print("=" * 68)
    print("MISE À JOUR DES MODÈLES ORANGE MONEY — 628 UTILISATEURS")
    print("=" * 68)

    # Sauvegarder les anciens artefacts avant toute mise à jour.
    sauvegarder_anciens_artefacts()

    df, X, y = charger_dataset()

    # 1. Évaluation non biaisée des 5 algorithmes.
    synthese, synthese_importance = evaluer_modeles(X, y)

    # 2. Modèles finaux destinés à être chargés par Flask.
    pipelines_finaux, parametres_finaux = entrainer_modeles_finaux(X, y)

    # 3. Clustering recalculé sur la base complète.
    df_final, score_silhouette = recalculer_segmentation(df)

    # 4. Background et métadonnées pour SHAP Logistic Regression.
    preparer_shap_et_metadonnees(
        df_final, X, pipelines_finaux, parametres_finaux, score_silhouette
    )

    print("\n" + "=" * 68)
    print("ENTRAÎNEMENT TERMINÉ")
    print("Les modèles finaux, K-Means, le dataset et les métriques sont sauvegardés.")
    print("Ne déploie pas encore : ml_loader.py et app.py doivent être adaptés aux pipelines.")
    print("=" * 68)


if __name__ == "__main__":
    main()
