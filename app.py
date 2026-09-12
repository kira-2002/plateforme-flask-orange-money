"""
Plateforme décisionnelle Orange Money — Version Flask
Étape 2+ : chargement des vrais modèles ML, scaler, K-Means, dataset
+ système de dataset actif par session (persiste sur toutes les pages)
"""

import os
import uuid
import pandas as pd
from flask import Flask, render_template, request, session, redirect, url_for
from ml_loader import charger_tout

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-key-a-changer-en-production')

# ══════════════════════════════════════════════════════════════
# CHARGEMENT UNIQUE AU DÉMARRAGE (pas à chaque requête !)
# ══════════════════════════════════════════════════════════════
DONNEES = charger_tout()

DOSSIER_UPLOADS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'uploads')
os.makedirs(DOSSIER_UPLOADS, exist_ok=True)

COLONNES_REQUISES_DATASET = ['fidelite', 'cluster', 'sat_globale', 'perception_frais']


def obtenir_dataset_actif():
    """Retourne le dataset actuellement actif pour cette session
    utilisateur : le dataset uploadé s'il y en a un de valide en
    session, sinon le dataset par défaut chargé au démarrage."""
    token = session.get('dataset_token')
    if token:
        chemin = os.path.join(DOSSIER_UPLOADS, f'{token}.csv')
        if os.path.exists(chemin):
            return pd.read_csv(chemin)
    return DONNEES['dataset']


def definir_dataset_actif(df, nom_fichier):
    """Enregistre un nouveau dataset comme actif pour cette session
    (remplace l'ancien s'il y en avait un)."""
    token = str(uuid.uuid4())
    chemin = os.path.join(DOSSIER_UPLOADS, f'{token}.csv')
    df.to_csv(chemin, index=False)
    session['dataset_token'] = token
    session['dataset_nom'] = nom_fichier


@app.context_processor
def injecter_dataset_actif():
    """Rend le nom du dataset personnalisé actif disponible dans
    TOUS les templates automatiquement (pour le badge dans la navbar)."""
    return {'dataset_actif_nom': session.get('dataset_nom')}


@app.route('/reinitialiser-dataset')
def reinitialiser_dataset():
    """Revient au dataset par défaut (supprime le dataset personnalisé
    de cette session)."""
    token = session.pop('dataset_token', None)
    session.pop('dataset_nom', None)
    if token:
        chemin = os.path.join(DOSSIER_UPLOADS, f'{token}.csv')
        if os.path.exists(chemin):
            os.remove(chemin)
    return redirect(request.referrer or url_for('accueil'))


def stats_accueil():
    """Calcule les indicateurs de la page d'accueil à partir du
    dataset actuellement actif (par défaut ou uploadé)."""
    df = obtenir_dataset_actif()
    return {
        'nb_utilisateurs': len(df),
        'taux_fidelite': round(df['fidelite'].mean() * 100, 1),
        'nb_segments': DONNEES['metadonnees']['k_optimal'],
        'nb_churners': int((df['fidelite'] == 0).sum()),
    }


MODULES = [
    {'icone': 'bi-bar-chart-fill', 'titre': 'Tableau de bord', 'description': "Indicateurs clés en un coup d'œil", 'url': '/dashboard'},
    {'icone': 'bi-magic', 'titre': 'Prédiction individuelle', 'description': 'Tester un profil utilisateur', 'url': '/prediction'},
    {'icone': 'bi-diagram-3-fill', 'titre': 'Segmentation K-Means', 'description': '5 profils comportementaux', 'url': '/segmentation'},
    {'icone': 'bi-search', 'titre': 'Analyse des facteurs', 'description': 'Ce qui influence la fidélité', 'url': '/facteurs'},
    {'icone': 'bi-lightbulb-fill', 'titre': 'Recommandations', 'description': 'Actions marketing par segment', 'url': '/recommandations'},
    {'icone': 'bi-exclamation-triangle-fill', 'titre': 'Alertes clients à risque', 'description': 'Surveillance proactive', 'url': '/alertes'},
]


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.route('/', methods=['GET', 'POST'])
def accueil():
    message_erreur = None
    message_succes = None

    if request.method == 'POST':
        fichier = request.files.get('fichier_dataset')
        if fichier and fichier.filename:
            try:
                df_uploade = pd.read_csv(fichier)
                manquantes = [c for c in COLONNES_REQUISES_DATASET if c not in df_uploade.columns]
                if manquantes:
                    message_erreur = f"Colonnes manquantes dans le fichier : {', '.join(manquantes)}. L'ancien dataset reste actif."
                else:
                    definir_dataset_actif(df_uploade, fichier.filename)
                    message_succes = f"Dataset personnalisé chargé : {fichier.filename} ({len(df_uploade)} lignes) — actif sur toute la plateforme."
            except Exception as e:
                message_erreur = f"Impossible de lire le fichier : {e}"

    return render_template(
        'index.html', stats=stats_accueil(), modules=MODULES,
        message_erreur=message_erreur, message_succes=message_succes,
    )


PALETTE_SEGMENTS = {
    'Utilisateurs à Risque': '#C62828',
    'Fidèles Engagés': '#2E7D32',
    'Captifs Insatisfaits': '#FF7900',
    'Fidèles Passifs': '#1565C0',
    'Nouveaux Utilisateurs': '#6A1B9A',
}


def calculer_donnees_dashboard(df):
    """Calcule tous les agrégats nécessaires au tableau de bord à
    partir d'un DataFrame donné (dataset par défaut ou uploadé)."""
    noms_clusters = DONNEES['metadonnees']['noms_clusters']
    df = df.copy()
    df['segment'] = df['cluster'].astype(str).map(noms_clusters)

    repartition_fid = {
        'labels': ['Fidèle', 'Churner'],
        'data': [int((df['fidelite'] == 1).sum()), int((df['fidelite'] == 0).sum())],
        'colors': ['#2E7D32', '#C62828'],
    }

    compte_segments = df['segment'].value_counts()
    segments_labels = list(compte_segments.index)
    repartition_segments = {
        'labels': segments_labels,
        'data': [int(v) for v in compte_segments.values],
        'colors': [PALETTE_SEGMENTS.get(s, '#999999') for s in segments_labels],
    }

    taux = (df.groupby('segment')['fidelite'].mean() * 100).sort_values()
    taux_segments = {
        'labels': list(taux.index),
        'data': [round(float(v), 1) for v in taux.values],
        'colors': ['#C62828' if v < 85 else '#2E7D32' for v in taux.values],
    }

    valeurs_sat = sorted(df['sat_globale'].unique().tolist())
    sat_fid = df[df['fidelite'] == 1]['sat_globale'].value_counts()
    sat_chu = df[df['fidelite'] == 0]['sat_globale'].value_counts()
    distribution_satisfaction = {
        'labels': [str(v) for v in valeurs_sat],
        'fideles': [int(sat_fid.get(v, 0)) for v in valeurs_sat],
        'churners': [int(sat_chu.get(v, 0)) for v in valeurs_sat],
    }

    valeurs_pf = sorted(df['perception_frais'].unique().tolist())
    pf_fid = df[df['fidelite'] == 1]['perception_frais'].value_counts()
    pf_chu = df[df['fidelite'] == 0]['perception_frais'].value_counts()
    distribution_frais = {
        'labels': [str(v) for v in valeurs_pf],
        'fideles': [int(pf_fid.get(v, 0)) for v in valeurs_pf],
        'churners': [int(pf_chu.get(v, 0)) for v in valeurs_pf],
    }

    stats = {
        'nb_utilisateurs': len(df),
        'taux_fidelite': round(df['fidelite'].mean() * 100, 1),
        'nb_segments': DONNEES['metadonnees']['k_optimal'],
        'nb_churners': int((df['fidelite'] == 0).sum()),
    }

    return {
        'stats': stats,
        'satisfaction_moyenne': round(float(df['sat_globale'].mean()), 2),
        'repartition_fid': repartition_fid,
        'repartition_segments': repartition_segments,
        'taux_segments': taux_segments,
        'distribution_satisfaction': distribution_satisfaction,
        'distribution_frais': distribution_frais,
    }


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    message_erreur = None
    message_succes = None

    if request.method == 'POST':
        fichier = request.files.get('fichier_dataset')
        if fichier and fichier.filename:
            try:
                df_uploade = pd.read_csv(fichier)
                manquantes = [c for c in COLONNES_REQUISES_DATASET if c not in df_uploade.columns]
                if manquantes:
                    message_erreur = f"Colonnes manquantes dans le fichier : {', '.join(manquantes)}. L'ancien dataset reste actif."
                else:
                    definir_dataset_actif(df_uploade, fichier.filename)
                    message_succes = f"Dataset personnalisé chargé : {fichier.filename} ({len(df_uploade)} lignes) — actif sur toute la plateforme."
            except Exception as e:
                message_erreur = f"Impossible de lire le fichier : {e}"

    df_actif = obtenir_dataset_actif()
    donnees_vue = calculer_donnees_dashboard(df_actif)
    return render_template(
        'dashboard.html',
        message_erreur=message_erreur,
        message_succes=message_succes,
        **donnees_vue
    )

import numpy as np
import shap

# ══════════════════════════════════════════════════════════════
# SHAP : explainer créé UNE SEULE FOIS au démarrage (coûteux à
# recréer à chaque requête)
# ══════════════════════════════════════════════════════════════
EXPLAINER_SHAP = shap.TreeExplainer(DONNEES['modeles']['Random Forest'])
COLONNES_CLUSTERING = ['sat_globale', 'frequence_usage', 'anciennete_usage', 'perception_frais']


@app.route('/prediction')
def prediction():
    return render_template('prediction.html')


@app.route('/api/predire', methods=['POST'])
def api_predire():
    from flask import request, jsonify
    import pandas as pd

    donnees_form = request.get_json()

    colonnes_modele = DONNEES['metadonnees']['colonnes_modele']
    cols_num = DONNEES['metadonnees']['colonnes_num_scaler']
    noms_clusters = DONNEES['metadonnees']['noms_clusters']

    # ── Construction du vecteur d'entrée (0 par défaut pour les
    # colonnes secondaires non couvertes par le formulaire) ──────
    entree = {col: 0 for col in colonnes_modele}
    entree.update({
        'tranche_age': int(donnees_form['tranche_age']),
        'anciennete_usage': int(donnees_form['anciennete_usage']),
        'frequence_usage': int(donnees_form['frequence_usage']),
        'sat_globale': int(donnees_form['sat_globale']),
        'perception_frais': int(donnees_form['perception_frais']),
        'utilise_orange_money': int(donnees_form['utilise_orange_money']),
        'wave_usage': int(donnees_form['wave_usage']),
        'presence_probleme': int(donnees_form['presence_probleme']),
        'motif_principal': int(donnees_form['motif_principal']),
    })

    X_utilisateur = pd.DataFrame([entree])[colonnes_modele]
    X_utilisateur_sc = X_utilisateur.copy()
    X_utilisateur_sc[cols_num] = DONNEES['scaler'].transform(X_utilisateur[cols_num])

    # ── Prédiction (Régression Logistique, meilleur AUC-ROC) ────
    modele_principal = DONNEES['modeles']['Régression Logistique']
    proba_fidele = float(modele_principal.predict_proba(X_utilisateur_sc)[0, 1])
    proba_churn = 1 - proba_fidele
    prediction_label = "Fidèle" if proba_fidele >= 0.5 else "Churner (à risque)"

    if proba_churn > 0.7:
        niveau_risque = "eleve"
    elif proba_churn > 0.4:
        niveau_risque = "modere"
    else:
        niveau_risque = "faible"

    # ── Segment K-Means ────────────────────────────────────────
    X_cluster = X_utilisateur[COLONNES_CLUSTERING].copy()
    X_cluster[COLONNES_CLUSTERING] = DONNEES['scaler_cluster'].transform(X_cluster)
    cluster_utilisateur = int(DONNEES['kmeans'].predict(X_cluster)[0])
    nom_cluster = noms_clusters[str(cluster_utilisateur)]

    # ── Explication SHAP individuelle ───────────────────────────
    shap_brut = EXPLAINER_SHAP.shap_values(X_utilisateur_sc)
    if isinstance(shap_brut, list):
        shap_val = shap_brut[1][0]
    elif isinstance(shap_brut, np.ndarray) and shap_brut.ndim == 3:
        shap_val = shap_brut[0, :, 1]
    else:
        shap_val = shap_brut[0]

    contributions = sorted(
        zip(colonnes_modele, [float(v) for v in shap_val]),
        key=lambda x: abs(x[1]), reverse=True
    )[:8]
    contributions = sorted(contributions, key=lambda x: x[1])  # tri pour affichage barre

    return jsonify({
        'prediction': prediction_label,
        'proba_fidele': round(proba_fidele * 100, 1),
        'proba_churn': round(proba_churn * 100, 1),
        'niveau_risque': niveau_risque,
        'segment': nom_cluster,
        'shap_labels': [c[0] for c in contributions],
        'shap_valeurs': [round(c[1], 4) for c in contributions],
    })

COLONNES_RADAR = ['sat_globale', 'frequence_usage', 'anciennete_usage', 'perception_frais', 'fidelite']
LABELS_RADAR = ['Satisfaction', "Fréquence d'usage", 'Ancienneté', 'Perception des frais', 'Taux de fidélité']


@app.route('/segmentation')
def segmentation():
    df = obtenir_dataset_actif().copy()
    noms_clusters = DONNEES['metadonnees']['noms_clusters']
    df['segment'] = df['cluster'].astype(str).map(noms_clusters)

    # ── Cartes de synthèse par segment ──────────────────────────
    cartes_segments = []
    for cluster_id in sorted(df['cluster'].dropna().unique()):
        nom = noms_clusters.get(str(int(cluster_id)), f'Cluster {int(cluster_id)}')
        sous_df = df[df['cluster'] == cluster_id]
        cartes_segments.append({
            'nom': nom,
            'taille': len(sous_df),
            'pct': round(len(sous_df) / len(df) * 100, 1),
            'taux_fidelite': round(sous_df['fidelite'].mean() * 100, 1),
            'couleur': PALETTE_SEGMENTS.get(nom, '#999999'),
        })

    # ── Profils moyens, normalisés pour le radar ────────────────
    profils = df.groupby('segment')[COLONNES_RADAR].mean()
    etendue = profils.max() - profils.min()
    profils_norm = ((profils - profils.min()) / etendue.replace(0, 1)).fillna(0.5)

    radar_datasets = []
    for segment in profils_norm.index:
        radar_datasets.append({
            'label': segment,
            'data': [round(float(v), 3) for v in profils_norm.loc[segment].values],
            'couleur': PALETTE_SEGMENTS.get(segment, '#999999'),
        })

    # ── Tableau détaillé (valeurs réelles, non normalisées) ─────
    tableau_profils = []
    for segment in profils.index:
        tableau_profils.append({
            'segment': segment,
            'satisfaction': round(float(profils.loc[segment, 'sat_globale']), 2),
            'frequence': round(float(profils.loc[segment, 'frequence_usage']), 2),
            'anciennete': round(float(profils.loc[segment, 'anciennete_usage']), 2),
            'frais': round(float(profils.loc[segment, 'perception_frais']), 2),
            'fidelite_pct': round(float(profils.loc[segment, 'fidelite']) * 100, 1),
        })

    return render_template(
        'segmentation.html',
        cartes_segments=cartes_segments,
        radar_labels=LABELS_RADAR,
        radar_datasets=radar_datasets,
        tableau_profils=tableau_profils,
        liste_segments=list(profils.index),
    )


@app.route('/api/segment/<segment_nom>')
def api_segment_detail(segment_nom):
    from flask import jsonify

    df = obtenir_dataset_actif().copy()
    noms_clusters = DONNEES['metadonnees']['noms_clusters']
    df['segment'] = df['cluster'].astype(str).map(noms_clusters)
    sous_df = df[df['segment'] == segment_nom]

    if len(sous_df) == 0:
        return jsonify({'erreur': 'Segment introuvable ou vide dans ce dataset'}), 404

    valeurs_sat = sorted(df['sat_globale'].dropna().unique().tolist())
    sat_fid = sous_df[sous_df['fidelite'] == 1]['sat_globale'].value_counts()
    sat_chu = sous_df[sous_df['fidelite'] == 0]['sat_globale'].value_counts()

    return jsonify({
        'nb_utilisateurs': len(sous_df),
        'taux_fidelite': round(float(sous_df['fidelite'].mean()) * 100, 1),
        'satisfaction_moyenne': round(float(sous_df['sat_globale'].mean()), 2),
        'frequence_moyenne': round(float(sous_df['frequence_usage'].mean()), 2),
        'sat_labels': [str(v) for v in valeurs_sat],
        'sat_fideles': [int(sat_fid.get(v, 0)) for v in valeurs_sat],
        'sat_churners': [int(sat_chu.get(v, 0)) for v in valeurs_sat],
    })

def couleur_correlation(val):
    """Dégradé rouge (corrélation négative) - blanc (nulle) - vert
    (positive), pour la heatmap de corrélation en CSS pur."""
    val = max(-1.0, min(1.0, val))
    if val >= 0:
        r = int(255 - val * (255 - 46))
        g = int(255 - val * (255 - 125))
        b = int(255 - val * (255 - 50))
    else:
        v = -val
        r = int(255 - v * (255 - 198))
        g = int(255 - v * (255 - 40))
        b = int(255 - v * (255 - 40))
    return f'rgb({r},{g},{b})'


@app.route('/facteurs')
def facteurs():
    df = obtenir_dataset_actif().copy()
    colonnes_modele = DONNEES['metadonnees']['colonnes_modele']
    cols_num = DONNEES['metadonnees']['colonnes_num_scaler']

    # ── Préparation de X dans l'ordre exact attendu par le modèle ──
    X = pd.DataFrame(index=df.index)
    for c in colonnes_modele:
        X[c] = df[c] if c in df.columns else 0
    X_sc = X.copy()
    X_sc[cols_num] = DONNEES['scaler'].transform(X[cols_num])

    # ── SHAP global ──────────────────────────────────────────────
    shap_brut = EXPLAINER_SHAP.shap_values(X_sc)
    if isinstance(shap_brut, list):
        shap_vals = shap_brut[1]
    elif isinstance(shap_brut, np.ndarray) and shap_brut.ndim == 3:
        shap_vals = shap_brut[:, :, 1]
    else:
        shap_vals = shap_brut

    importance_shap = np.abs(shap_vals).mean(axis=0)
    ordre = np.argsort(importance_shap)[::-1][:15]
    shap_labels = [colonnes_modele[i] for i in ordre][::-1]
    shap_valeurs = [round(float(importance_shap[i]), 4) for i in ordre][::-1]

    # ── Importance Random Forest (Gini) ─────────────────────────
    rf = DONNEES['modeles']['Random Forest']
    importances_rf = rf.feature_importances_
    mediane_rf = float(np.median(importances_rf))
    ordre_rf = np.argsort(importances_rf)[::-1][:15]
    rf_labels = [colonnes_modele[i] for i in ordre_rf][::-1]
    rf_valeurs = [round(float(importances_rf[i]), 4) for i in ordre_rf][::-1]
    rf_couleurs = [('#2E7D32' if importances_rf[i] >= mediane_rf else '#C62828') for i in ordre_rf][::-1]

    # ── Corrélations de Spearman (heatmap en CSS) ───────────────
    variables_disponibles = ['sat_globale', 'perception_frais', 'frequence_usage',
                              'anciennete_usage', 'tranche_age', 'fidelite']
    labels_fr = {'sat_globale': 'Satisfaction', 'perception_frais': 'Frais',
                 'frequence_usage': 'Fréquence', 'anciennete_usage': 'Ancienneté',
                 'tranche_age': 'Âge', 'fidelite': 'Fidélité'}
    variables_presentes = [v for v in variables_disponibles if v in df.columns]
    matrice = df[variables_presentes].corr(method='spearman')

    heatmap = []
    for v1 in variables_presentes:
        cellules = []
        for v2 in variables_presentes:
            val = float(matrice.loc[v1, v2])
            cellules.append({'valeur': round(val, 2), 'couleur': couleur_correlation(val)})
        heatmap.append({'label': labels_fr[v1], 'cellules': cellules})
    labels_colonnes_corr = [labels_fr[v] for v in variables_presentes]

    return render_template(
        'facteurs.html',
        shap_labels=shap_labels, shap_valeurs=shap_valeurs,
        rf_labels=rf_labels, rf_valeurs=rf_valeurs, rf_couleurs=rf_couleurs,
        heatmap=heatmap, labels_colonnes_corr=labels_colonnes_corr,
    )

RECOMMANDATIONS = {
    'Fidèles Engagés': {
        'icone': 'bi-star-fill',
        'diagnostic': "Satisfaction, usage et fidélité élevés — le segment le plus précieux.",
        'objectif': "Capitaliser sur leur engagement et en faire des ambassadeurs.",
        'actions': [
            "Programme VIP avec avantages exclusifs",
            "Système de parrainage récompensé",
            "Accès prioritaire aux nouveaux services financiers",
            "Sollicitation pour témoignages / avis clients",
        ],
    },
    'Fidèles Passifs': {
        'icone': 'bi-moon-fill',
        'diagnostic': "Fidèles mais peu actifs — une fidélité \"dormante\" plutôt qu'engagée.",
        'objectif': "Réactiver l'usage sans forcer, pour transformer la fidélité passive en usage actif.",
        'actions': [
            "Notifications ciblées sur les nouveaux services",
            "Rappels d'usage discrets (pas de sur-sollicitation)",
            "Mise en avant d'offres adaptées à un usage occasionnel",
            "Enquête légère pour comprendre le frein à l'usage régulier",
        ],
    },
    'Captifs Insatisfaits': {
        'icone': 'bi-lightning-charge-fill',
        'diagnostic': "Usage fréquent MAIS satisfaction très faible — un profil à risque de départ malgré une fidélité apparente.",
        'objectif': "Traiter en priorité l'insatisfaction avant qu'elle ne se traduise en churn.",
        'actions': [
            "Amélioration prioritaire du service client",
            "Geste commercial ciblé (remise, offre de compensation)",
            "Enquête de satisfaction approfondie pour identifier la cause",
            "Suivi personnalisé rapproché",
        ],
    },
    'Nouveaux Utilisateurs': {
        'icone': 'bi-flower1',
        'diagnostic': "Ancienneté très faible — une relation encore en construction.",
        'objectif': "Ancrer durablement l'habitude d'usage dès les premiers mois.",
        'actions': [
            "Parcours d'accueil renforcé (onboarding)",
            "Tutoriels d'utilisation des principales fonctionnalités",
            "Offres de bienvenue pour encourager le premier usage",
            "Point de contact dédié pendant les 3 premiers mois",
        ],
    },
    'Utilisateurs à Risque': {
        'icone': 'bi-exclamation-triangle-fill',
        'diagnostic': "Satisfaction faible, usage rare, perception négative des frais — le taux de fidélité le plus bas de tous les segments.",
        'objectif': "Rétention prioritaire avant un désengagement définitif.",
        'actions': [
            "Intervention proactive de rétention",
            "Réduction tarifaire ciblée",
            "Contact prioritaire avec le service client",
            "Suivi rapproché sur les 30 prochains jours",
        ],
    },
}


@app.route('/recommandations')
def recommandations():
    df = obtenir_dataset_actif().copy()
    noms_clusters = DONNEES['metadonnees']['noms_clusters']
    df['segment'] = df['cluster'].astype(str).map(noms_clusters)

    segments_info = []
    for nom_segment, contenu in RECOMMANDATIONS.items():
        sous_df = df[df['segment'] == nom_segment]
        segments_info.append({
            'nom': nom_segment,
            'icone': contenu['icone'],
            'diagnostic': contenu['diagnostic'],
            'objectif': contenu['objectif'],
            'actions': contenu['actions'],
            'taille': len(sous_df),
            'taux_fidelite': round(float(sous_df['fidelite'].mean()) * 100, 1) if len(sous_df) else 0,
        })

    return render_template(
        'recommandations.html',
        segments_info=segments_info,
        recommandations_json=RECOMMANDATIONS,
    )

@app.route('/alertes')
def alertes():
    df = obtenir_dataset_actif().copy()
    colonnes_modele = DONNEES['metadonnees']['colonnes_modele']
    cols_num = DONNEES['metadonnees']['colonnes_num_scaler']
    noms_clusters = DONNEES['metadonnees']['noms_clusters']

    # ── Préparation de X et calcul du score de risque pour tous ──
    X = pd.DataFrame(index=df.index)
    for c in colonnes_modele:
        X[c] = df[c] if c in df.columns else 0
    X_sc = X.copy()
    X_sc[cols_num] = DONNEES['scaler'].transform(X[cols_num])

    modele = DONNEES['modeles']['Régression Logistique']
    proba_fidele = modele.predict_proba(X_sc)[:, 1]
    proba_churn = 1 - proba_fidele

    df_resultat = df.copy()
    df_resultat['ID'] = df_resultat.index
    df_resultat['Score_risque'] = (proba_churn * 100).round(1)
    df_resultat['Segment'] = df_resultat['cluster'].astype(str).map(noms_clusters)

    def niveau(score):
        if score > 70:
            return 'eleve'
        elif score > 40:
            return 'modere'
        return 'faible'
    df_resultat['Niveau'] = df_resultat['Score_risque'].apply(niveau)
    df_resultat = df_resultat.sort_values('Score_risque', ascending=False)

    nb_eleve = int((df_resultat['Niveau'] == 'eleve').sum())
    nb_modere = int((df_resultat['Niveau'] == 'modere').sum())
    nb_faible = int((df_resultat['Niveau'] == 'faible').sum())

    # ── Alerte prioritaire : Captifs Insatisfaits en alerte rouge ──
    captifs_risque = df_resultat[(df_resultat['Segment'] == 'Captifs Insatisfaits') & (df_resultat['Niveau'] == 'eleve')]

    lignes = []
    for _, row in df_resultat.iterrows():
        lignes.append({
            'id': int(row['ID']),
            'score': float(row['Score_risque']),
            'niveau': row['Niveau'],
            'segment': row['Segment'],
            'satisfaction': int(row['sat_globale']) if pd.notna(row['sat_globale']) else None,
            'frequence': int(row['frequence_usage']) if pd.notna(row['frequence_usage']) else None,
            'anciennete': int(row['anciennete_usage']) if pd.notna(row['anciennete_usage']) else None,
            'frais': int(row['perception_frais']) if pd.notna(row['perception_frais']) else None,
        })

    return render_template(
        'alertes.html',
        nb_total=len(df_resultat), nb_eleve=nb_eleve, nb_modere=nb_modere, nb_faible=nb_faible,
        nb_captifs_risque=len(captifs_risque),
        liste_segments=list(noms_clusters.values()),
        lignes=lignes,
    )


if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'True') == 'True'
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=debug_mode, host='0.0.0.0', port=port)