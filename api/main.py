"""
API FastAPI : entraînement, évaluation et prédiction.

L'application envoie sa configuration (données, variables, prétraitement, modèles, hyperparamètres) à POST /train.
L'entraînement s'exécute en tâche de fond (pas de dépassement de délai) : on suit l'avancement avec GET /jobs/{id}.
Les modèles entraînés sont gardés en mémoire dans une « session » (5 sessions maximum).

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

from src import models as M
from src import trainer as T

MAX_ROWS = 100_000
MAX_SESSIONS = 5
MAX_JOBS = 30

app = FastAPI(
    title="API de classification : entraînement, évaluation, prédiction",
    version="2.0.0",
    description=("Service utilisé par l'application web : **entraîne** plusieurs modèles de classification "
                 "(régression logistique, arbre, KNN, Random Forest, XGBoost) avec le prétraitement et les "
                 "hyperparamètres choisis, **évalue** (AUC, PR-AUC, précision, rappel, F1, Gini, KS, matrice de "
                 "confusion à un seuil réglable) et **prédit** pour un nouvel individu.\n\n"
                 "Flux : `POST /train` → `GET /jobs/{job_id}` (jusqu'à `done`) → `GET /sessions/{id}/metrics` "
                 "et `POST /sessions/{id}/predict`."),
)

_lock = threading.Lock()
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


def _run_job(job_id, cfg, df):
    def progress(i, n, msg):
        with _lock:
            JOBS[job_id].update(progress=(i / n) if n else 0.0, message=msg)

    try:
        session = T.run_training(cfg, df, progress=progress, prepared=True)
        sid = uuid.uuid4().hex[:12]
        summary = T.session_summary(session)
        summary["session_id"] = sid
        with _lock:
            SESSIONS[sid] = session
            _trim(SESSIONS, MAX_SESSIONS)
            JOBS[job_id].update(status="done", progress=1.0, message="Terminé", result=summary)
    except Exception as e:  # erreur d'entraînement : renvoyée à l'application
        with _lock:
            JOBS[job_id].update(status="error", error=f"{type(e).__name__}: {e}", message="Échec")


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health", tags=["Service"], summary="État du service")
def health():
    return {"status": "ok", "sessions": len(SESSIONS), "jobs_running": sum(j["status"] == "running" for j in JOBS.values())}


@app.get("/catalog", tags=["Service"], summary="Modèles, hyperparamètres par défaut et grilles de recherche")
def catalog():
    return {"models": M.MODEL_NAMES, "default_params": M.DEFAULT_PARAMS, "search_grids": M.PARAM_GRIDS}


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
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        JOBS[job_id] = {"job_id": job_id, "status": "running", "progress": 0.0, "message": "Démarrage…",
                        "result": None, "error": None, "created": time.time()}
        _trim(JOBS, MAX_JOBS)
    threading.Thread(target=_run_job, args=(job_id, cfg, df), daemon=True).start()
    return {"job_id": job_id, "status": "running"}


@app.get("/jobs/{job_id}", tags=["Entraînement"], summary="Avancement et résultat d'un entraînement")
def job_status(job_id: str):
    with _lock:
        j = JOBS.get(job_id)
        if j is None:
            raise HTTPException(404, "Tâche inconnue.")
        return dict(j)


@app.get("/sessions/{session_id}", tags=["Sessions"], summary="Résumé d'une session entraînée")
def session_info(session_id: str):
    return T.session_summary(_get_session(session_id))


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
