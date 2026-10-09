# 📊 Classification interactive : Telco Customer Churn

Projet Challenge 1 · Data Mining · ENSI 2026-2027 · Cas 13 (Télécom, attrition client)

Application web interactive : l'utilisateur charge les données (téléchargement automatique ou CSV), choisit la cible et les
variables, règle le prétraitement, choisit les modèles et leurs hyperparamètres, **entraîne** (via l'API FastAPI),
**compare** 5 modèles, **évalue** (ROC, précision-rappel, matrice de confusion, seuil réglable) et prédit pour un nouvel individu.

| | Lien |
|---|---|
| Application déployée | _à compléter_ |
| API (Swagger `/docs`) | _à compléter_ |
| One-pager | `docs/one_pager.pdf` |

## Captures d'écran
_À ajouter (dossier `docs/`)_ : onglets Données, Modélisation, Évaluation (ROC, seuil), Prédiction.

## Architecture
```
Application Streamlit  ──(JSON)──►  API FastAPI
  données, préparation              POST /train  → tâche de fond (prétraitement + 5 modèles + hyperparamètres)
  choix des modèles                 GET  /jobs/{id}                → avancement, scores du jeu de test
  évaluation interactive            GET  /sessions/{id}/metrics    → métriques à un seuil
  (ROC, PR, confusion, seuil)       POST /sessions/{id}/predict    → prédiction d'un individu
```
L'API renvoie les **scores du jeu de test** : l'application calcule alors métriques, courbes et matrice de confusion en direct,
sans appel réseau quand on bouge le curseur de seuil. Si l'API est injoignable, l'application entraîne **localement**
avec le même code (`src/trainer.py`).

## Fonctionnalités
| Module | Contenu | Où |
|---|---|---|
| 1. Données | téléchargement automatique (kagglehub) ou CSV importé, aperçu, statistiques, manquants, classes, graphiques interactifs | onglet 1, `src/eda.py` |
| 2. Préparation | cible/variables, imputation, encodage, normalisation, split paramétrable | onglet 2, `src/preprocess.py` |
| 3. Modélisation | régression logistique, arbre, KNN, Random Forest, XGBoost ; hyperparamètres (recherche automatique ou manuels) | onglet 3, `src/models.py`, API |
| 4. Évaluation | AUC, PR-AUC, précision, rappel, F1, Gini, KS ; ROC superposées, PR, confusion, curseur de seuil | onglet 4, `src/plots.py` |
| API | entraînement, évaluation, prédiction | `api/main.py`, `src/trainer.py` |

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

Les sessions sont gardées **en mémoire** (5 maximum) : un redémarrage du service les efface.

## Structure
```
app/streamlit_app.py     application
api/main.py              API FastAPI
src/                     data, eda, preprocess, models, plots, trainer, api_client
notebooks/               01_exploration, 02_modelisation_evaluation
docs/                    déploiement, modèle de one-pager, captures
```

## Choix méthodologiques
- Split **stratifié** ; prétraitement ajusté sur le train uniquement (`Pipeline`) ; validation croisée avec ré-ajustement par pli.
- Déséquilibre (73/27) : `class_weight="balanced"` / `scale_pos_weight` ; métriques AUC, PR-AUC, rappel, F1 ; seuil réglable.
- Les variables sensibles (genre, senior, couple, personnes à charge) sont conservées dans le jeu de variables.

## Déploiement
Voir `docs/DEPLOIEMENT.md`.
