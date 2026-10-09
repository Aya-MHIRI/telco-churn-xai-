# Déploiement

Ordre : **1) API → 2) application**. Aucun modèle n'est à commiter : l'API entraîne à la demande.

## 0. Fixer les versions
Pour que local et en ligne se comportent pareil :
```bash
pip freeze | grep -i -E "^(pandas|numpy|scikit-learn|xgboost|plotly|streamlit|fastapi|uvicorn|requests|kagglehub)=="
```
Copiez ces lignes (avec `==`) dans `requirements.txt` ; reprenez pandas, numpy, scikit-learn, xgboost, fastapi, uvicorn
dans `requirements-api.txt`. Utilisez la même version de Python (3.11 recommandé) partout.

## 1. API sur Render (le plus simple)
1. render.com → **New → Web Service** → connecter le dépôt GitHub.
2. Runtime : Python 3. **Build command** : `pip install -r requirements-api.txt`
3. **Start command** : `uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1`
4. **Environment** : `TRAIN_N_JOBS=1` (l'instance gratuite a peu de mémoire : la recherche d'hyperparamètres
   s'exécute alors sans processus parallèles).
5. Une fois déployé : `https://VOTRE-API.onrender.com/docs` (Swagger) et `/health`.

Points d'attention (instance gratuite) :
- Elle **s'endort** après inactivité : faites un appel à `/health` avant de présenter (réveil ~1 min).
- Au réveil, les **sessions en mémoire sont perdues** : l'application affiche alors « Session introuvable »
  pour la prédiction ; il suffit de ré-entraîner.
- Elle est lente : pour une démonstration, décochez « Optimiser les hyperparamètres » ou réduisez les combinaisons.
  L'entraînement est une tâche de fond avec suivi (`/jobs/{id}`), donc pas de dépassement de délai HTTP.
- **Un seul worker** (`--workers 1`) : l'état est en mémoire.

### Alternative : Hugging Face Spaces (Docker)
1. huggingface.co → **New Space** → SDK **Docker**.
2. Poussez le dépôt (le `Dockerfile` écoute sur le port 7860).
3. Dans le `README.md` du Space, l'en-tête YAML doit contenir `sdk: docker` et `app_port: 7860`.

## 2. Application sur Streamlit Community Cloud
1. share.streamlit.io → **New app** → dépôt GitHub, branche `main`.
2. **Main file path** : `app/streamlit_app.py` ; Python 3.11 dans *Advanced settings*.
3. *Advanced settings → Secrets* :
```toml
API_URL = "https://VOTRE-API.onrender.com"
```
4. Commitez le CSV de secours dans `data/` : si le téléchargement automatique échoue, l'appli l'utilise.

Si l'API est injoignable au moment d'entraîner, l'application bascule automatiquement en **entraînement local**
(un avertissement s'affiche).

## 3. Vérifications finales
- [ ] `/docs` de l'API s'affiche
- [ ] Dans l'appli : onglet 3, « Entraîner via l'API FastAPI » activé → message « Modèles entraînés (API FastAPI) »
- [ ] Onglet 5 : la prédiction indique « calculé par : API FastAPI »
- [ ] Liens et captures d'écran ajoutés au README
