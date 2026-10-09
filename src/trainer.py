"""
Entraînement et prédiction côté « service ».
Utilisé par l'API FastAPI (entraînement en tâche de fond) et, en secours, directement par l'application.
"""
import numpy as np
import pandas as pd

from src import data as D
from src import models as M
from src import preprocess as P

DEFAULT_CFG = {
    "target": "Churn", "positive_label": 1, "features": None, "test_size": 0.2, "imputer": "median",
    "scale": True, "simplify": True, "engineer": True, "seed": 42,
    "models": list(M.MODEL_NAMES), "tune": True, "n_iter": 10, "cv": 3, "params": {},
}


def _py(o):
    """Convertit récursivement numpy/pandas en types JSON (NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_py(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return _py(o.to_dict(orient="records"))
    if isinstance(o, np.ndarray):
        return _py(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if o is pd.NA or o is None:
        return None
    return o


def build_schema(df, raw_features):
    """Description des variables d'entrée (types, modalités, bornes) : sert aux formulaires de prédiction."""
    schema = {}
    for c in raw_features:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            schema[c] = {"type": "num", "min": float(s.min()), "max": float(s.max()), "median": float(s.median())}
        else:
            vc = s.dropna().astype(str).value_counts()
            schema[c] = {"type": "cat", "choices": sorted(vc.index.tolist()), "mode": str(vc.index[0])}
    return schema


# --------------------------------------------------------------------------- #
# Entraînement
# --------------------------------------------------------------------------- #
def prepare_dataframe(cfg, data):
    """Données brutes (liste de dicts ou DataFrame) -> DataFrame propre avec cible 0/1."""
    df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    target = cfg["target"]
    if target not in df.columns:
        raise ValueError(f"Variable cible '{target}' absente des données.")
    df = D.generic_clean(df)
    df = df.dropna(subset=[target]).copy()
    df[target] = P.encode_target(df[target], cfg.get("positive_label")).astype(int)
    return df


def validate_config(cfg, df):
    target = cfg["target"]
    if not cfg["models"]:
        raise ValueError("Aucun modèle sélectionné.")
    unknown = [m for m in cfg["models"] if m not in M.MODEL_NAMES]
    if unknown:
        raise ValueError(f"Modèle(s) inconnu(s) : {unknown}. Disponibles : {M.MODEL_NAMES}")
    feats = cfg.get("features") or [c for c in df.columns if c != target]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise ValueError(f"Variables absentes des données : {missing}")
    if target in feats:
        raise ValueError("La cible ne peut pas faire partie des variables explicatives.")
    if not feats:
        raise ValueError("Aucune variable explicative.")
    counts = df[target].value_counts()
    if len(counts) != 2 or counts.min() < 10:
        raise ValueError("La cible doit avoir 2 classes avec au moins 10 observations chacune.")
    if len(df) < 100:
        raise ValueError("Au moins 100 observations sont nécessaires.")


def run_training(cfg, data, progress=None, prepared=False):
    """
    Prépare, entraîne les modèles choisis et renvoie une « session » (modèles + scores sur le jeu de test).
    progress(i, n, message) : rappel optionnel pour afficher l'avancement.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    df = data if prepared else prepare_dataframe(cfg, data)
    validate_config(cfg, df)
    target = cfg["target"]
    features = cfg["features"] or [c for c in df.columns if c != target]

    prep = P.prepare(df, target, features, test_size=cfg["test_size"], imputer=cfg["imputer"], scale=cfg["scale"],
                     simplify=cfg["simplify"], engineer=cfg["engineer"], seed=cfg["seed"])
    names = list(cfg["models"])
    pipes, best_params, cv_scores = {}, {}, {}
    for i, name in enumerate(names):
        if progress:
            progress(i, len(names), f"Entraînement de {name}…")
        pipe, bp, cv_auc = M.train_model(name, prep["preprocessor"], prep["X_train"], prep["y_train"],
                                         tune=cfg["tune"], params=cfg["params"].get(name), n_iter=cfg["n_iter"],
                                         cv=cfg["cv"], seed=cfg["seed"])
        pipes[name], best_params[name], cv_scores[name] = pipe, bp, cv_auc
    if progress:
        progress(len(names), len(names), "Évaluation sur le jeu de test…")

    return {
        "pipelines": pipes, "target": target, "cfg": cfg,
        "raw_features": prep["raw_features"], "features": prep["features"],
        "num_cols": prep["num_cols"], "cat_cols": prep["cat_cols"], "settings": prep["settings"],
        "feature_names": prep["feature_names"], "schema": build_schema(df, prep["raw_features"]),
        "y_test": prep["y_test"].to_numpy(), "scores": M.score_models(pipes, prep["X_test"]),
        "best_params": best_params, "cv_scores": cv_scores,
        "n_train": len(prep["X_train"]), "n_test": len(prep["X_test"]),
        "pos_rate_train": float(prep["y_train"].mean()), "pos_rate_test": float(prep["y_test"].mean()),
    }


def session_summary(session):
    """Version JSON de la session (envoyée à l'application) : scores de test, paramètres, schéma."""
    return _py({
        "models": list(session["pipelines"]), "target": session["target"],
        "n_train": session["n_train"], "n_test": session["n_test"],
        "pos_rate_train": session["pos_rate_train"], "pos_rate_test": session["pos_rate_test"],
        "num_cols": session["num_cols"], "cat_cols": session["cat_cols"],
        "raw_features": session["raw_features"], "feature_names": session["feature_names"],
        "schema": session["schema"], "best_params": session["best_params"], "cv_scores": session["cv_scores"],
        "y_test": session["y_test"].astype(int),
        "scores": {m: np.round(s, 5) for m, s in session["scores"].items()},
    })


def metrics_payload(session, threshold=0.5):
    """Métriques de tous les modèles à un seuil donné + matrices de confusion."""
    table = M.evaluate_scores(session["scores"], session["y_test"], threshold)
    conf = {m: M.confusion_counts(session["y_test"], s, threshold) for m, s in session["scores"].items()}
    return _py({"threshold": threshold, "metrics": table.reset_index().rename(columns={"index": "Modèle"}),
                "confusion": conf, "best_model": table.index[0]})


# --------------------------------------------------------------------------- #
# Prédiction d'un individu
# --------------------------------------------------------------------------- #
def to_features_frame(session, data):
    """Dict (variables brutes) -> DataFrame d'une ligne avec les colonnes attendues par le pipeline."""
    if isinstance(data, pd.Series):
        data = data.to_dict()
    raw, schema = session["raw_features"], session["schema"]
    df = pd.DataFrame([{c: data.get(c, np.nan) for c in raw}])
    for c in raw:
        if schema[c]["type"] == "num":
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = df[c].astype("object")
    s = session["settings"]
    df = P.engineer_input(df, s["simplify"], s["engineer"])
    return df[session["features"]]


def predict_one(session, model, data, threshold=0.5):
    if model not in session["pipelines"]:
        raise KeyError(f"Modèle inconnu '{model}'. Disponibles : {list(session['pipelines'])}")
    X = to_features_frame(session, data)
    proba = float(session["pipelines"][model].predict_proba(X)[0, 1])
    decision = int(proba >= threshold)
    return _py({"model": model, "probability": proba, "threshold": float(threshold), "decision": decision,
                "decision_label": "classe positive prédite" if decision else "classe négative prédite",
                "missing_features": [c for c in session["raw_features"] if c not in data or data[c] is None]})
