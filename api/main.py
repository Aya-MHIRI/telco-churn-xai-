"""
API FastAPI : entraînement, évaluation, prédiction et EXPLICABILITÉ (SHAP, LIME).

L'application envoie sa configuration (données, variables, prétraitement, modèles, hyperparamètres) à POST /train.
Les calculs longs (entraînement, importance par permutation, SHAP global) s'exécutent en tâche de fond :
on suit l'avancement avec GET /jobs/{id}. Les modèles entraînés sont gardés en mémoire dans une « session »
(5 sessions maximum).

Lancement :  uvicorn api.main:app --reload --port 8000      (un seul worker : l'état est en mémoire)
Documentation interactive (Swagger) : http://localhost:8000/docs
"""
import os
import sys
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Literal, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src import explain_service as X
from src import models as M
from src import trainer as T

MAX_ROWS = 100_000
MAX_SESSIONS = 5
MAX_JOBS = 30

app = FastAPI(
    title="API de classification : entraînement, évaluation, prédiction, explicabilité",
    version="3.0.0",
    description=("Service utilisé par l'application web : **entraîne** plusieurs modèles de classification "
                 "(régression logistique, arbre, KNN, Random Forest, XGBoost), **évalue** (AUC, PR-AUC, précision, "
                 "rappel, F1, Gini, KS), **prédit** et **explique** les décisions (importances, SHAP, LIME, "
                 "comparaison SHAP/LIME, phrase de synthèse, équité).\n\n"
                 "Flux : `POST /train` → `GET /jobs/{job_id}` (jusqu'à `done`) → `session_id` → routes "
                 "`/sessions/{id}/...`. Les explications globales (`explain/permutation`, `explain/shap-global`) "
                 "sont aussi des tâches de fond."),
)

_lock = threading.Lock()
_heavy = threading.Semaphore(1)     # un seul calcul lourd à la fois (limite la mémoire sur un petit hébergeur)
SESSIONS: "OrderedDict[str, dict]" = OrderedDict()
JOBS: "OrderedDict[str, dict]" = OrderedDict()


# --------------------------------------------------------------------------- #
# Schémas
# --------------------------------------------------------------------------- #
class TrainRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Lignes du jeu de données (liste d'objets colonne -> valeur)")
    target: str = Field(..., description="Nom de la variable cible")
    positive_label: Any = Field(1, description="Valeur de la classe positive (ignorée si la cible est déjà 0/1)")
    features: Optional[List[str]] = Field(None, description="Variables explicatives (défaut : toutes)")
    test_size: float = Field(0.2, ge=0.05, le=0.5, description="Part du jeu de test (split stratifié)")
    imputer: Literal["median", "mean", "zero"] = "median"
    scale: bool = Field(True, description="Normalisation StandardScaler")
    simplify: bool = Field(True, description="Regrouper « No internet/phone service » en « No »")
    engineer: bool = Field(True, description="Créer nb_services / has_protection (jeu Telco)")
    seed: int = 42
    models: List[str] = Field(default_factory=lambda: list(M.MODEL_NAMES))
    tune: bool = Field(True, description="Optimiser les hyperparamètres (RandomizedSearchCV, score AUC)")
    n_iter: int = Field(10, ge=1, le=50, description="Combinaisons testées par modèle si tune")
    cv: int = Field(3, ge=2, le=10, description="Plis de validation croisée si tune")
    params: Dict[str, Dict[str, Any]] = Field(default_factory=dict,
                                              description="Hyperparamètres par modèle si tune=false, "
                                                          "ex. {\"KNN\": {\"n_neighbors\": 9}}")


EXAMPLE_CUSTOMER = {
    "model": "XGBoost", "threshold": 0.5,
    "data": {"gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 2,
             "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
             "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
             "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
             "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
             "MonthlyCharges": 85.5, "TotalCharges": 171.0},
}


class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE_CUSTOMER})
    model: str = Field("XGBoost", description="Nom d'un modèle de la session")
    threshold: float = Field(0.5, ge=0, le=1, description="Seuil de décision")
    data: Dict[str, Any] = Field(..., description="Variables brutes de l'individu (voir GET /sessions/{id})")


class ShapGlobalRequest(BaseModel):
    model: str = Field("XGBoost", description="Modèle à expliquer")
    n: int = Field(300, ge=20, le=500, description="Nombre d'individus du jeu de test (60 max pour KNN)")


class PermutationRequest(BaseModel):
    n_repeats: int = Field(5, ge=1, le=10, description="Répétitions du mélange de chaque variable")
    n_max: int = Field(500, ge=100, le=2000, description="Individus du jeu de test utilisés (borne le temps de calcul)")


# --------------------------------------------------------------------------- #
# Utilitaires
# --------------------------------------------------------------------------- #
def _trim(store, limit):
    while len(store) > limit:
        store.popitem(last=False)


def _get_session(sid):
    with _lock:
        s = SESSIONS.get(sid)
    if s is None:
        raise HTTPException(404, "Session introuvable ou expirée (le service a pu redémarrer) : relancez l'entraînement.")
    return s


def _spawn(fn, kind):
    """Lance fn(progress) en tâche de fond et renvoie l'identifiant de la tâche."""
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        JOBS[job_id] = {"job_id": job_id, "kind": kind, "status": "running", "progress": 0.0,
                        "message": "Démarrage…", "result": None, "error": None, "created": time.time()}
        _trim(JOBS, MAX_JOBS)

    def progress(i, n, msg):
        with _lock:
            JOBS[job_id].update(progress=(i / n) if n else 0.0, message=msg)

    def runner():
        with _lock:
            JOBS[job_id]["message"] = "En attente d'un calcul en cours…"
        with _heavy:
            with _lock:
                JOBS[job_id]["message"] = "Démarrage…"
            try:
                result = fn(progress)
                with _lock:
                    JOBS[job_id].update(status="done", progress=1.0, message="Terminé", result=result)
            except Exception as e:  # erreur de calcul : renvoyée à l'application
                with _lock:
                    JOBS[job_id].update(status="error", error=f"{type(e).__name__}: {e}", message="Échec")

    threading.Thread(target=runner, daemon=True).start()
    return job_id


def _guard(fn, *a, **kw):
    """Exécute un calcul d'explication en traduisant les erreurs en réponses HTTP lisibles."""
    try:
        return fn(*a, **kw)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\""))
    except IndexError as e:
        raise HTTPException(404, str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
@app.get("/health", tags=["Service"], summary="État du service")
def health():
    return {"status": "ok", "sessions": len(SESSIONS), "jobs_running": sum(j["status"] == "running" for j in JOBS.values())}


@app.get("/catalog", tags=["Service"], summary="Modèles, hyperparamètres par défaut et grilles de recherche")
def catalog():
    return {"models": M.MODEL_NAMES, "default_params": M.DEFAULT_PARAMS, "search_grids": M.PARAM_GRIDS}


# --------------------------------------------------------------------------- #
# Entraînement
# --------------------------------------------------------------------------- #
@app.post("/train", status_code=202, tags=["Entraînement"], summary="Lancer l'entraînement (tâche de fond)")
def train(req: TrainRequest):
    if len(req.data) > MAX_ROWS:
        raise HTTPException(413, f"Trop de lignes ({len(req.data)} > {MAX_ROWS}).")
    cfg = req.model_dump(exclude={"data"})
    try:
        df = T.prepare_dataframe(cfg, req.data)
        T.validate_config(cfg, df)
    except ValueError as e:
        raise HTTPException(422, str(e))

    def work(progress):
        session = T.run_training(cfg, df, progress=progress, prepared=True)
        sid = uuid.uuid4().hex[:12]
        summary = T.session_summary(session)
        summary["session_id"] = sid
        with _lock:
            SESSIONS[sid] = session
            _trim(SESSIONS, MAX_SESSIONS)
        return summary

    return {"job_id": _spawn(work, "train"), "status": "running"}


@app.get("/jobs/{job_id}", tags=["Entraînement"], summary="Avancement et résultat d'une tâche de fond")
def job_status(job_id: str):
    with _lock:
        j = JOBS.get(job_id)
        if j is None:
            raise HTTPException(404, "Tâche inconnue.")
        return dict(j)


# --------------------------------------------------------------------------- #
# Sessions, évaluation, prédiction
# --------------------------------------------------------------------------- #
@app.get("/sessions/{session_id}", tags=["Sessions"], summary="Résumé d'une session entraînée")
def session_info(session_id: str):
    return T.session_summary(_get_session(session_id))


@app.get("/sessions/{session_id}/individual/{index}", tags=["Sessions"],
         summary="Un individu du jeu de test (variables brutes et vraie classe)")
def individual(session_id: str, index: int):
    return _guard(X.get_individual, _get_session(session_id), index)


@app.get("/sessions/{session_id}/metrics", tags=["Évaluation"],
         summary="Métriques de tous les modèles à un seuil donné")
def session_metrics(session_id: str, threshold: float = 0.5):
    if not 0 <= threshold <= 1:
        raise HTTPException(422, "threshold doit être entre 0 et 1.")
    return T.metrics_payload(_get_session(session_id), threshold)


@app.post("/sessions/{session_id}/predict", tags=["Prédiction"], summary="Probabilité et décision pour un individu")
def predict(session_id: str, req: PredictRequest):
    session = _get_session(session_id)
    try:
        return T.predict_one(session, req.model, req.data, req.threshold)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\""))
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# Module 5 : explicabilité
# --------------------------------------------------------------------------- #
@app.get("/sessions/{session_id}/explain/native", tags=["Explications globales"],
         summary="Importance propre au modèle (coefficients, importance par impureté)")
def explain_native(session_id: str, model: str):
    return _guard(X.native, _get_session(session_id), model)


@app.post("/sessions/{session_id}/explain/permutation", status_code=202, tags=["Explications globales"],
          summary="Importance par permutation de tous les modèles (tâche de fond)")
def explain_permutation(session_id: str, req: PermutationRequest = PermutationRequest()):
    session = _get_session(session_id)
    return {"job_id": _spawn(lambda p: X.permutation(session, req.n_repeats, req.n_max, p), "permutation"),
            "status": "running"}


@app.post("/sessions/{session_id}/explain/shap-global", status_code=202, tags=["Explications globales"],
          summary="Valeurs SHAP sur un échantillon du jeu de test : summary plot et dépendances (tâche de fond)")
def explain_shap_global(session_id: str, req: ShapGlobalRequest):
    session = _get_session(session_id)
    if req.model not in session["pipelines"]:
        raise HTTPException(404, f"Modèle inconnu '{req.model}'. Disponibles : {list(session['pipelines'])}")
    return {"job_id": _spawn(lambda p: X.shap_global(session, req.model, req.n, p), "shap-global"),
            "status": "running"}


@app.post("/sessions/{session_id}/explain/shap", tags=["Explications locales"],
          summary="Explication SHAP (waterfall) d'un individu")
def explain_shap(session_id: str, req: PredictRequest, top_k: int = 10):
    return _guard(X.shap_local, _get_session(session_id), req.model, req.data, req.threshold, top_k)


@app.post("/sessions/{session_id}/explain/lime", tags=["Explications locales"],
          summary="Explication LIME d'un individu")
def explain_lime(session_id: str, req: PredictRequest, num_features: int = 10):
    return _guard(X.lime_local, _get_session(session_id), req.model, req.data, req.threshold, num_features)


@app.post("/sessions/{session_id}/explain/compare", tags=["Explications locales"],
          summary="SHAP + LIME + cohérence + phrase de synthèse")
def explain_compare(session_id: str, req: PredictRequest, top_k: int = 10):
    return _guard(X.compare_local, _get_session(session_id), req.model, req.data, req.threshold, top_k)


@app.get("/sessions/{session_id}/fairness", tags=["Équité"],
         summary="Prédictions par groupe d'une variable sensible (parité, rappel, faux positifs)")
def fairness(session_id: str, model: str, threshold: float = 0.5, sensitive: str = ""):
    cols = [c.strip() for c in sensitive.split(",") if c.strip()]
    return _guard(X.fairness, _get_session(session_id), model, threshold, cols)
