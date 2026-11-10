"""
Plateforme décisionnelle Orange Money — Version Flask
Étape 2+ : chargement des vrais modèles ML, scaler, K-Means, dataset
+ pipelines à prétraitement intégré, dataset 628 et segmentation actualisée
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

COLONNES_REQUISES_DATASET = [
    'tranche_age', 'sexe', 'situation_professionnelle',
    'anciennete_usage', 'frequence_usage', 'sat_globale',
    'perception_frais', 'presence_probleme', 'operateur_prefere',
    'motif_principal', 'pb_aucun', 'pb_autre', 'pb_lenteur',
    'pb_probleme_de_reseau', 'pb_service_client',
    'pb_transaction_echouee', 'serv_achat_de_credit',
    'serv_paiement_de_factures', 'serv_paiement_marchand',
    'serv_reception_dargent', 'serv_transfert_dargent',
    'fidelite', 'cluster',
]


def valider_dataset_uploade(df):
    """Vérifie le schéma du CSV préparé utilisé par la plateforme."""
    if df.empty:
        return "Le fichier CSV est vide."

    manquantes = [c for c in COLONNES_REQUISES_DATASET if c not in df.columns]
    if manquantes:
        return (
            "Colonnes manquantes dans le fichier : "
            f"{', '.join(manquantes)}. Le fichier doit contenir les 23 colonnes "
            "du dataset préparé. Le dataset actif précédent est conservé."
        )

    if df[COLONNES_REQUISES_DATASET].isna().any().any():
        return (
            "Le fichier contient des valeurs manquantes dans les 23 colonnes "
            "requises. Nettoyez le CSV avant de l'importer. "
            "Le dataset actif précédent est conservé."
        )

    if not df['fidelite'].isin([0, 1]).all():
        return "La colonne 'fidelite' doit contenir uniquement 0 et 1."

    if not df['cluster'].isin([0, 1, 2, 3, 4]).all():
        return "La colonne 'cluster' doit contenir les identifiants 0 à 4."

    return None


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

    if request.method == 'POST':
        fichier = request.files.get('fichier_dataset')
        if not fichier or not fichier.filename:
            message_erreur = "Sélectionnez un fichier CSV avant de cliquer sur Charger et explorer."
        else:
            try:
                df_uploade = pd.read_csv(fichier)
                message_erreur = valider_dataset_uploade(df_uploade)
                if message_erreur is None:
                    definir_dataset_actif(df_uploade, fichier.filename)
                    session['message_upload_succes'] = (
                        f"Dataset chargé : {fichier.filename} ({len(df_uploade)} lignes). "
                        "Il est maintenant utilisé dans toute la plateforme."
                    )
                    # Après un import valide, ouvrir directement le tableau de bord.
                    return redirect(url_for('dashboard'))
            except Exception as e:
                message_erreur = f"Impossible de lire le fichier : {e}"

    return render_template(
        'index.html', stats=stats_accueil(), modules=MODULES,
        message_erreur=message_erreur, message_succes=None,
    )


PALETTE_SEGMENTS = {
    'Utilisateurs à Risque': '#C62828',
    'Fidèles Engagés': '#2E7D32',
    'Utilisateurs Captifs Insatisfaits': '#FF7900',
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
    message_succes = session.pop('message_upload_succes', None)

    if request.method == 'POST':
        fichier = request.files.get('fichier_dataset')
        if not fichier or not fichier.filename:
            message_erreur = "Sélectionnez un fichier CSV avant de lancer le chargement."
        else:
            try:
                df_uploade = pd.read_csv(fichier)
                message_erreur = valider_dataset_uploade(df_uploade)
                if message_erreur is None:
                    definir_dataset_actif(df_uploade, fichier.filename)
                    session['message_upload_succes'] = (
                        f"Dataset chargé : {fichier.filename} ({len(df_uploade)} lignes). "
                        "Il est maintenant utilisé dans toute la plateforme."
                    )
                    return redirect(url_for('dashboard'))
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
from scipy import sparse

# ══════════════════════════════════════════════════════════════
# PRÉTRAITEMENT ET SHAP — RÉGRESSION LOGISTIQUE UNIQUEMENT
# Le pipeline contient déjà StandardScaler + OneHotEncoder.
# Le background SHAP a été produit avec le préprocesseur final.
# ══════════════════════════════════════════════════════════════
PIPELINE_LOGISTIQUE = DONNEES['modeles']['Régression Logistique']
PREPROCESSEUR_LOGISTIQUE = PIPELINE_LOGISTIQUE.named_steps['preprocesseur']
CLASSIFIEUR_LOGISTIQUE = PIPELINE_LOGISTIQUE.named_steps['clf']
BACKGROUND_SHAP = np.asarray(DONNEES['shap_background'], dtype=float)
EXPLAINER_SHAP = shap.LinearExplainer(CLASSIFIEUR_LOGISTIQUE, BACKGROUND_SHAP)

COLONNES_MODELE = DONNEES['metadonnees']['colonnes_modele']
COLONNES_CATEGORIELLES = DONNEES['metadonnees']['colonnes_categorielles']
COLONNES_NUMERIQUES = DONNEES['metadonnees']['colonnes_num_scaler']
COLONNES_CLUSTERING = DONNEES['metadonnees']['colonnes_clustering']
VALEURS_REFERENCE = DONNEES['metadonnees'].get('valeurs_reference_prediction', {})
NOMS_FEATURES_SHAP = DONNEES['metadonnees'].get(
    'variables_transforme_shap', PREPROCESSEUR_LOGISTIQUE.get_feature_names_out().tolist()
)

MAPPING_MOTIF_FORMULAIRE = {
    '0': 'Disponibilité des points',
    '1': 'Fiabilité',
    '2': 'Habitude',
    '3': 'Promotions',
    '4': 'Réseau',
}


def preparer_X_modele(df):
    """Construit les 21 variables sources attendues par les pipelines.

    Pour un dataset personnalisé incomplet, les variables absentes sont
    complétées avec les valeurs de référence calculées à l'entraînement.
    """
    X = pd.DataFrame(index=df.index)
    for colonne in COLONNES_MODELE:
        valeur_ref = VALEURS_REFERENCE.get(colonne, 0)
        if colonne in df.columns:
            X[colonne] = df[colonne]
            X[colonne] = X[colonne].where(X[colonne].notna(), valeur_ref)
        else:
            X[colonne] = valeur_ref

    for colonne in COLONNES_CATEGORIELLES:
        X[colonne] = X[colonne].astype(str).str.strip()

    for colonne in COLONNES_MODELE:
        if colonne not in COLONNES_CATEGORIELLES:
            X[colonne] = pd.to_numeric(X[colonne], errors='coerce')
            X[colonne] = X[colonne].fillna(VALEURS_REFERENCE.get(colonne, 0))

    return X[COLONNES_MODELE]


def transformer_pour_shap(X):
    matrice = PREPROCESSEUR_LOGISTIQUE.transform(X)
    if sparse.issparse(matrice):
        matrice = matrice.toarray()
    return np.asarray(matrice, dtype=float)


def extraire_shap_classe_fidele(matrice_transformee):
    """Retourne les valeurs SHAP de la classe 1 (fidèles déclarés)."""
    valeurs = EXPLAINER_SHAP.shap_values(matrice_transformee)
    if isinstance(valeurs, list):
        valeurs = valeurs[1] if len(valeurs) > 1 else valeurs[0]
    valeurs = np.asarray(valeurs)
    if valeurs.ndim == 3:
        if valeurs.shape[-1] == 2:
            valeurs = valeurs[:, :, 1]
        elif valeurs.shape[0] == 2:
            valeurs = valeurs[1]
        else:
            raise ValueError(f"Format SHAP inattendu : {valeurs.shape}")
    if valeurs.ndim != 2:
        raise ValueError(f"Format SHAP inattendu : {valeurs.shape}")
    return valeurs


def regrouper_shap_par_variable(valeurs_shap):
    """Regroupe les modalités One-Hot en contributions par variable source."""
    mapping = []
    for nom in NOMS_FEATURES_SHAP:
        nom = str(nom)
        if '__' in nom:
            transformateur, variable_transformee = nom.split('__', 1)
        else:
            transformateur, variable_transformee = '', nom

        if transformateur == 'nominales':
            source = next(
                (c for c in COLONNES_CATEGORIELLES
                 if variable_transformee.startswith(c + '_')),
                variable_transformee
            )
        else:
            source = variable_transformee
        mapping.append(source)

    resultat = np.zeros((valeurs_shap.shape[0], len(COLONNES_MODELE)), dtype=float)
    for j, variable in enumerate(COLONNES_MODELE):
        indices = [i for i, source in enumerate(mapping) if source == variable]
        if indices:
            resultat[:, j] = valeurs_shap[:, indices].sum(axis=1)
    return resultat


def construire_entree_prediction(donnees_form):
    """Construit le profil du formulaire dans le format brut du pipeline."""
    entree = dict(VALEURS_REFERENCE)

    champs_ordinaux = [
        'tranche_age', 'anciennete_usage', 'frequence_usage',
        'sat_globale', 'perception_frais'
    ]
    for champ in champs_ordinaux:
        entree[champ] = int(donnees_form.get(champ, entree.get(champ, 1)))

    motif = str(donnees_form.get('motif_principal', '1'))
    entree['motif_principal'] = MAPPING_MOTIF_FORMULAIRE.get(motif, motif)
    entree['presence_probleme'] = int(donnees_form.get('presence_probleme', 0))

    colonnes_pb = [c for c in COLONNES_MODELE if c.startswith('pb_')]
    if entree['presence_probleme'] == 0:
        for c in colonnes_pb:
            entree[c] = 1 if c == 'pb_aucun' else 0
    else:
        entree['pb_aucun'] = 0
        for c in colonnes_pb:
            if c != 'pb_aucun':
                entree[c] = int(VALEURS_REFERENCE.get(c, 0))

    # Les cases « utilise Orange Money » et « utilise Wave » sont conservées
    # dans l'interface historique, mais ne font pas partie des 21 prédicteurs.
    X = pd.DataFrame([{c: entree.get(c, VALEURS_REFERENCE.get(c, 0))
                       for c in COLONNES_MODELE}], columns=COLONNES_MODELE)
    for c in COLONNES_CATEGORIELLES:
        X[c] = X[c].astype(str).str.strip()
    for c in COLONNES_MODELE:
        if c not in COLONNES_CATEGORIELLES:
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)
    return X


@app.route('/prediction')
def prediction():
    return render_template('prediction.html')


@app.route('/api/predire', methods=['POST'])
def api_predire():
    from flask import jsonify

    donnees_form = request.get_json(silent=True) or {}
    try:
        X_utilisateur = construire_entree_prediction(donnees_form)

        # Le pipeline effectue lui-même standardisation et One-Hot Encoding.
        modele_principal = DONNEES['modeles']['Régression Logistique']
        probabilites = modele_principal.predict_proba(X_utilisateur)[0]
        classes = list(modele_principal.classes_)
        proba_fidele = float(probabilites[classes.index(1)])
        proba_churn = 1.0 - proba_fidele
        classe_predite = int(modele_principal.predict(X_utilisateur)[0])
        prediction_label = 'Fidèle' if classe_predite == 1 else 'Churner (à risque)'

        if proba_churn > 0.7:
            niveau_risque = 'eleve'
        elif proba_churn > 0.4:
            niveau_risque = 'modere'
        else:
            niveau_risque = 'faible'

        # Affectation K-Means avec le scaler propre à la segmentation.
        X_cluster = X_utilisateur[COLONNES_CLUSTERING].astype(float)
        X_cluster_sc = DONNEES['scaler_cluster'].transform(X_cluster)
        cluster_utilisateur = int(DONNEES['kmeans'].predict(X_cluster_sc)[0])
        nom_cluster = DONNEES['metadonnees']['noms_clusters'][str(cluster_utilisateur)]

        # SHAP de la régression logistique, regroupé sur les 21 variables sources.
        X_transforme = transformer_pour_shap(X_utilisateur)
        shap_brut = extraire_shap_classe_fidele(X_transforme)
        shap_groupe = regrouper_shap_par_variable(shap_brut)[0]
        contributions = sorted(
            zip(COLONNES_MODELE, [float(v) for v in shap_groupe]),
            key=lambda x: abs(x[1]), reverse=True
        )[:8]
        contributions = sorted(contributions, key=lambda x: x[1])

        return jsonify({
            'prediction': prediction_label,
            'proba_fidele': round(proba_fidele * 100, 1),
            'proba_churn': round(proba_churn * 100, 1),
            'niveau_risque': niveau_risque,
            'segment': nom_cluster,
            'shap_labels': [c[0] for c in contributions],
            'shap_valeurs': [round(c[1], 4) for c in contributions],
            'message_interpretation': 'Contributions SHAP de la régression logistique vers la fidélité déclarée (classe 1).',
        })
    except Exception as erreur:
        app.logger.exception('Erreur lors de la prédiction individuelle')
        return jsonify({'erreur': f'Impossible de calculer la prédiction : {erreur}'}), 400

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
    X = preparer_X_modele(df)

    # SHAP global : régression logistique, puis regroupement des modalités OHE.
    X_transforme = transformer_pour_shap(X)
    shap_brut = extraire_shap_classe_fidele(X_transforme)
    shap_groupe = regrouper_shap_par_variable(shap_brut)
    importance_shap = np.abs(shap_groupe).mean(axis=0)
    ordre = np.argsort(importance_shap)[::-1][:15]
    shap_labels = [COLONNES_MODELE[i] for i in ordre][::-1]
    shap_valeurs = [round(float(importance_shap[i]), 4) for i in ordre][::-1]

    # Importance par permutation de la régression logistique (F1 de la classe 0),
    # moyenne calculée sur les 50 plis externes de validation croisée.
    importance_df = DONNEES['importance_permutation'].copy()
    importance_df = importance_df.sort_values('importance_moyenne', ascending=False).head(15)
    permutation_labels = importance_df['variable'].astype(str).tolist()[::-1]
    permutation_valeurs = [round(float(v), 4) for v in importance_df['importance_moyenne'].tolist()][::-1]
    permutation_couleurs = [
        '#2E7D32' if v >= 0 else '#C62828'
        for v in importance_df['importance_moyenne'].tolist()
    ][::-1]

    # Corrélations de Spearman conservées comme dans la version précédente.
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
        permutation_labels=permutation_labels,
        permutation_valeurs=permutation_valeurs,
        permutation_couleurs=permutation_couleurs,
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
    'Utilisateurs Captifs Insatisfaits': {
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
    noms_clusters = DONNEES['metadonnees']['noms_clusters']

    # Le pipeline de régression logistique applique lui-même le prétraitement.
    X = preparer_X_modele(df)
    modele = DONNEES['modeles']['Régression Logistique']
    probabilites = modele.predict_proba(X)
    classes = list(modele.classes_)
    proba_fidele = probabilites[:, classes.index(1)]
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
    captifs_risque = df_resultat[(df_resultat['Segment'] == 'Utilisateurs Captifs Insatisfaits') & (df_resultat['Niveau'] == 'eleve')]

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