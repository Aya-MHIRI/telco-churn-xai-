"""
Module 5 (explicabilité) côté « service » : fonctions qui travaillent sur une session entraînée.
Utilisées par l'API FastAPI (routes /sessions/{id}/explain/...) et, en secours, directement par l'application.
Toutes les fonctions renvoient des structures JSON (listes / dictionnaires).
"""
import numpy as np
import pandas as pd

from src import trainer as T
from src import xai

MAX_KNN_ROWS = 60       # SHAP par noyau (KNN) : lent, échantillon limité
MAX_SHAP_ROWS = 500


def _cache(session):
    return session.setdefault("_cache", {})


def _pipe(session, model):
    if model not in session["pipelines"]:
        raise KeyError(f"Modèle inconnu '{model}'. Disponibles : {list(session['pipelines'])}")
    return session["pipelines"][model]


def positive_name(session):
    return "départ" if session["target"] == "Churn" else f"« {session['target']} »"


# --------------------------------------------------------------------------- #
# Individu du jeu de test
# --------------------------------------------------------------------------- #
def get_individual(session, index):
    X = session["X_test"]
    if not 0 <= index < len(X):
        raise IndexError(f"index hors limites (0 à {len(X) - 1}).")
    row = X.iloc[index][session["raw_features"]].to_dict()
    return T._py({"index": int(index), "data": row, "true_label": int(session["y_test"][index]),
                  "n_test": len(X)})


# --------------------------------------------------------------------------- #
# Explications globales
# --------------------------------------------------------------------------- #
def native(session, model):
    ni = xai.native_importance(_pipe(session, model))
    return T._py({"model": model, "available": ni is not None, "importance": ni})


def permutation(session, n_repeats=5, n_max=500, progress=None):
    d = xai.permutation_all(session["pipelines"], session["X_test"], session["y_test"], n_repeats, n_max, progress)
    return T._py({"n_repeats": n_repeats, "n_rows": int(min(n_max, len(session["X_test"]))), "importance": d})


def shap_global(session, model, n=300, progress=None):
    pipe = _pipe(session, model)
    kernel = xai.model_kind(pipe.named_steps["model"]) == "kernel"
    n = int(min(n, MAX_KNN_ROWS if kernel else MAX_SHAP_ROWS, len(session["X_test"])))
    if progress:
        progress(0, 2, f"SHAP sur {n} individus ({model})…")
    sample = session["X_test"].sample(n, random_state=0)
    res = xai.shap_values(pipe, model, sample, session["background"], _cache(session))
    pre = pipe.named_steps["pre"]
    data = xai.unscaled_df(pre, res.data, session["num_cols"])
    imp = xai.shap_importance_by_feature(res.values, res.feature_names, session["num_cols"], session["cat_cols"])
    if progress:
        progress(2, 2, "Terminé")
    return T._py({
        "model": model, "n": n, "units": res.units, "base_value": float(np.mean(res.base)),
        "feature_names": res.feature_names,
        "values": np.round(res.values, 4), "data": np.round(data.values.astype(float), 4),
        "importance_by_variable": imp,
    })


# --------------------------------------------------------------------------- #
# Explications locales
# --------------------------------------------------------------------------- #
def _shap_core(session, model, data, threshold):
    pipe = _pipe(session, model)
    X = T.to_features_frame(session, data)
    res = xai.shap_values(pipe, model, X, session["background"], _cache(session))
    contrib = xai.contributions_df(res.values[0], res.feature_names, X.iloc[0], session["num_cols"], session["cat_cols"])
    proba = float(pipe.predict_proba(X)[0, 1])
    summary = xai.generate_summary(contrib, proba, threshold, positive_name(session))
    return X, pipe, res, contrib, proba, summary


def _shap_block(res, contrib, top_k):
    top = contrib.head(top_k)
    return {"base_value": float(res.base[0]), "units": res.units,
            "contributions": top[["feature", "label", "valeur", "contribution"]].rename(columns={"valeur": "value"}),
            "other_contribution": float(contrib.iloc[top_k:]["contribution"].sum()),
            "n_other_features": max(len(contrib) - top_k, 0)}


def shap_local(session, model, data, threshold=0.5, top_k=10):
    _, _, res, contrib, proba, summary = _shap_core(session, model, data, threshold)
    return T._py({"model": model, "probability": proba, "threshold": float(threshold),
                  "decision": int(proba >= threshold), "summary": summary, **_shap_block(res, contrib, top_k)})


def lime_local(session, model, data, threshold=0.5, num_features=10, num_samples=2000):
    pipe = _pipe(session, model)
    X = T.to_features_frame(session, data)
    ctx = xai.lime_context(pipe, model, session["lime_train"], session["num_cols"], _cache(session))
    res = xai.lime_explain(ctx, X, session["num_cols"], session["cat_cols"], num_features, num_samples)
    proba = float(pipe.predict_proba(X)[0, 1])
    return T._py({"model": model, "probability": proba, "threshold": float(threshold),
                  "rules": res["table"], "local_fit_r2": res["score"], "local_pred": res["local_pred"]})


def compare_local(session, model, data, threshold=0.5, top_k=10, k_compare=5, num_samples=2000):
    """SHAP + LIME + cohérence + phrase de synthèse, en un seul appel."""
    X, pipe, res, contrib, proba, summary = _shap_core(session, model, data, threshold)
    ctx = xai.lime_context(pipe, model, session["lime_train"], session["num_cols"], _cache(session))
    lime = xai.lime_explain(ctx, X, session["num_cols"], session["cat_cols"], top_k, num_samples)
    cmp_ = xai.compare_shap_lime(contrib, lime["table"], k_compare)
    return T._py({
        "model": model, "probability": proba, "threshold": float(threshold), "decision": int(proba >= threshold),
        "summary": summary, "shap": _shap_block(res, contrib, top_k),
        "lime": {"rules": lime["table"], "local_fit_r2": lime["score"]},
        "comparison": {k: cmp_[k] for k in ("k", "overlap", "overlap_pct", "common", "sign_agreement", "table")},
    })


# --------------------------------------------------------------------------- #
# Équité
# --------------------------------------------------------------------------- #
def fairness(session, model, threshold=0.5, sensitive=None):
    pipe = _pipe(session, model)
    sensitive = [c for c in (sensitive or []) if c in session["X_test"].columns]
    rep = xai.fairness_report(pipe, session["X_test"], session["y_test"], sensitive, threshold)
    return T._py({"model": model, "threshold": float(threshold), "report": rep, "gaps": xai.parity_gaps(rep)})
