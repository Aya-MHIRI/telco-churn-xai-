# Classification explicable : Telco Customer Churn

**Projet Challenge 1 · Data Mining · ENSI 2026-2027 · Cas 13 (Télécom, attrition client)**
Auteure : Aya MHIRI, II3 IA (travail individuel)

Application web interactive qui permet de charger les données, les préparer, entraîner et comparer 5 modèles de
classification, les évaluer (seuil de décision réglable) puis **expliquer** leurs décisions (importances, SHAP, LIME,
équité), globalement et client par client. L'entraînement et les explications sont assurés par une API FastAPI.

## Liens

| Ressource | Lien |
|---|---|
| Dépôt GitHub | https://github.com/Aya-MHIRI/telco-churn-xai-.git |
| Application Streamlit déployée | https://telco-churndata.streamlit.app/ |
| API FastAPI (documentation Swagger) |https://telco-churn-xai.onrender.com/docs |
| One-pager (PDF) | `docs/one_pager.pdf` |

> L'API gratuite peut mettre une trentaine de secondes à se réveiller après une période d'inactivité. Si elle est
> injoignable, l'application entraîne automatiquement en local avec le même code (`src/trainer.py`).

## Problème métier
Un opérateur télécom perd des clients (attrition, « churn »). Objectif : repérer à l'avance les clients susceptibles de
partir pour déclencher une action de rétention ciblée, et comprendre **pourquoi** le modèle les signale.

## Données
Telco Customer Churn (IBM / Kaggle) : 7043 clients, 20 variables après suppression de `customerID`, cible `Churn`.
- Classes déséquilibrées : 26,5 % de départs (73/27).
- `TotalCharges` est stockée en texte, avec 11 valeurs vides cachées (clients à `tenure = 0`).
- Départs par contrat : mensuel 42,7 %, un an 11,3 %, deux ans 2,8 %.

## Résultats (jeu de test de 1409 clients, seuil 0,5)

| Modèle | AUC-ROC | PR-AUC | Précision | Rappel | F1 |
|---|---|---|---|---|---|
| **XGBoost** | **0,8459** | **0,6584** | 0,513 | 0,805 | 0,626 |
| Random Forest | 0,8452 | 0,6572 | 0,522 | 0,797 | 0,631 |
| Régression logistique | 0,8409 | 0,6273 | 0,505 | 0,781 | 0,613 |
| Arbre de décision | 0,8361 | 0,6220 | 0,524 | 0,805 | 0,635 |
| KNN | 0,8304 | 0,6226 | 0,607 | 0,570 | 0,588 |

Meilleur modèle : **XGBoost**. Les 5 modèles sont proches (écart d'AUC de 0,015) : la régression logistique, bien plus
simple à expliquer, perd seulement 0,005 d'AUC.
Seuil optimal pour le F1 : **0,6** (précision 0,572, rappel 0,735, F1 0,643). Le seuil se règle dans l'application selon le
coût d'un départ raté face au coût d'une offre inutile.

## Explications clés (module 5)
- **Variables les plus influentes (SHAP, XGBoost)** : type de contrat (30 % de l'importance), service internet (16 %),
  ancienneté `tenure` (15 %), mode de paiement (7 %), charges mensuelles (6 %).
- **Effets** : contrat mensuel, fibre optique, paiement par chèque électronique et ancienneté faible (moins de 6 mois)
  augmentent le risque ; contrats longs, absence d'internet, services de sécurité/support et forte ancienneté le réduisent.
- **Cohérence SHAP / LIME** sur un client réellement parti (probabilité 60,7 %) : 4 variables communes sur 5 dans le
  top 5, 100 % d'accord sur le sens des effets.
- **Équité** : les variables sensibles (genre, senior, couple, personnes à charge) portent **2,4 %** de l'importance
  SHAP. Écart de taux de prédictions positives : genre 0,015 (négligeable), mais personnes à charge 0,276, partenaire
  0,254 et senior 0,302, qui reflètent surtout des taux réels de départ différents entre groupes.
- Les explications décrivent le modèle (associations), pas des causes. Guide de lecture : `docs/MODULE5.md`.

## Architecture
```
Application Streamlit  ──(JSON)──►  API FastAPI
  données, préparation              POST /train  → tâche de fond (prétraitement + 5 modèles + hyperparamètres)
  choix des modèles                 GET  /jobs/{id}                → avancement, scores du jeu de test
  évaluation interactive            GET  /sessions/{id}/metrics    → métriques à un seuil
  (ROC, PR, confusion, seuil)       POST /sessions/{id}/predict    → prédiction d'un individu
  explicabilité (SHAP, LIME)        .../explain/*  et /fairness    → explications globales et locales, équité
```
L'API renvoie les **scores du jeu de test** : l'application calcule alors métriques, courbes et matrice de confusion en
direct, sans appel réseau quand on bouge le curseur de seuil.

## Fonctionnalités par module
| Module | Contenu | Où |
|---|---|---|
| 1. Données | téléchargement automatique (kagglehub) ou CSV importé, aperçu, statistiques, manquants, classes, graphiques | onglet 1, `src/eda.py` |
| 2. Préparation | cible/variables, imputation, encodage, normalisation, split paramétrable | onglet 2, `src/preprocess.py` |
| 3. Modélisation | régression logistique, arbre, KNN, Random Forest, XGBoost ; hyperparamètres (recherche automatique ou manuels) | onglet 3, `src/models.py`, API |
| 4. Évaluation | AUC, PR-AUC, précision, rappel, F1, Gini, KS ; ROC, PR, confusion, seuil réglable | onglet 4, `src/plots.py` |
| 5. Explicabilité | importance native et par permutation, SHAP (beeswarm, dépendance, waterfall), LIME, comparaison, phrase de synthèse, équité | onglet 5, `src/xai.py`, `src/explain_service.py`, `src/xai_plots.py` |
| 6. API et déploiement | entraînement, évaluation, prédiction, explications | `api/main.py`, `src/trainer.py` |
| Prédiction | probabilité et décision pour un nouveau client | onglet 6 |

## Installation
Python 3.10 ou 3.11 recommandé.
```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux : source venv/bin/activate)
pip install -r requirements.txt
python test_setup.py             # attendu : (7043, 20) / ~0.265 / 11
```
Plan B si le téléchargement Kaggle échoue : placer `WA_Fn-UseC_-Telco-Customer-Churn.csv` dans `data/`.

## Utilisation
```bash
uvicorn api.main:app --port 8000             # terminal 1 : API  -> http://localhost:8000/docs
streamlit run app/streamlit_app.py           # terminal 2 : appli -> http://localhost:8501
```
L'adresse de l'API se règle par la variable `API_URL` (ou le secret Streamlit `API_URL`). Défaut : `http://localhost:8000`.

## API
| Méthode | Route | Rôle |
|---|---|---|
| GET | `/health` | état du service |
| GET | `/catalog` | modèles, hyperparamètres par défaut, grilles de recherche |
| POST | `/train` | lance l'entraînement (données, cible, variables, prétraitement, modèles, hyperparamètres) |
| GET | `/jobs/{job_id}` | avancement ; résultat (scores, paramètres retenus, `session_id`) |
| GET | `/sessions/{id}` | résumé de la session |
| GET | `/sessions/{id}/metrics?threshold=` | métriques et matrices de confusion à un seuil |
| POST | `/sessions/{id}/predict` | probabilité et décision pour un individu |
| GET | `/sessions/{id}/individual/{i}` | un individu du jeu de test (variables brutes, vraie classe) |
| GET | `/sessions/{id}/explain/native?model=` | importance propre au modèle |
| POST | `/sessions/{id}/explain/permutation` | importance par permutation, tous modèles (tâche de fond) |
| POST | `/sessions/{id}/explain/shap-global` | valeurs SHAP sur un échantillon (tâche de fond) |
| POST | `/sessions/{id}/explain/shap` · `/lime` · `/compare` | explication SHAP, LIME ou les deux, avec cohérence et phrase de synthèse |
| GET | `/sessions/{id}/fairness` | taux de prédictions positives par groupe d'une variable sensible |

Les sessions sont gardées **en mémoire** (5 maximum) : un redémarrage du service les efface.

## Structure du dépôt
```
app/streamlit_app.py     application
api/main.py              API FastAPI
src/                     data, eda, preprocess, models, plots, trainer, xai, explain_service, xai_plots, api_client
notebooks/               01_exploration, 02_modelisation_evaluation, 03_explicabilite
docs/                    déploiement, guide du module 5, one-pager
requirements.txt         dépendances de l'application
requirements-api.txt     dépendances de l'API
```

## Choix méthodologiques
- Split **stratifié** ; prétraitement ajusté sur le train uniquement (`Pipeline`) ; validation croisée avec ré-ajustement
  par pli (pas de fuite de données).
- Déséquilibre (73/27) : `class_weight="balanced"` / `scale_pos_weight` ; métriques AUC, PR-AUC, rappel, F1 ; seuil réglable.
- Variables sensibles conservées dans le modèle pour mesurer leur influence (onglet Équité).
- Les contributions SHAP des colonnes one-hot sont additionnées par variable d'origine dans les explications locales.

## Limites
- Association et non causalité ; un seul jeu de données et un seul découpage train/test.
- SHAP et LIME sont des approximations ; LIME peut varier d'un tirage à l'autre (R² local de 0,68 sur l'exemple étudié).
- Sessions en mémoire et API gratuite qui s'endort : adaptés à une démonstration, pas à de la production.

## Déploiement
Voir `docs/DEPLOIEMENT.md`.
