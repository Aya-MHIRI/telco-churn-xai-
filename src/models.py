"""Modélisation, optimisation des hyperparamètres et métriques (sans dépendance graphique)."""
import math
import os

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

MODEL_NAMES = ["Logistic Regression", "Decision Tree", "KNN", "Random Forest", "XGBoost"]

# Valeurs par défaut (utilisées sans optimisation)
DEFAULT_PARAMS = {
    "Logistic Regression": {"C": 1.0},
    "Decision Tree": {"max_depth": 5, "min_samples_leaf": 20},
    "KNN": {"n_neighbors": 15, "weights": "uniform"},
    "Random Forest": {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 5},
    "XGBoost": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
}

# Petites grilles (RandomizedSearchCV)
PARAM_GRIDS = {
    "Logistic Regression": {"C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
    "Decision Tree": {"max_depth": [3, 4, 5, 6, 8], "min_samples_leaf": [5, 10, 20, 50]},
    "KNN": {"n_neighbors": [5, 11, 15, 21, 31, 41], "weights": ["uniform", "distance"]},
    "Random Forest": {"n_estimators": [100, 200, 300], "max_depth": [5, 8, 12], "min_samples_leaf": [5, 10, 20]},
    "XGBoost": {"n_estimators": [100, 200, 300], "max_depth": [2, 3, 4, 5], "learning_rate": [0.03, 0.05, 0.1],
                "subsample": [0.8, 1.0], "colsample_bytree": [0.7, 0.9, 1.0]},
}

# Hyperparamètres éditables dans l'interface : (nom, type, min/options, max, aide)
PARAM_SPECS = {
    "Logistic Regression": [("C", "float", 0.001, 100.0, "Inverse de la régularisation")],
    "Decision Tree": [("max_depth", "int", 1, 20, "Profondeur maximale"),
                      ("min_samples_leaf", "int", 1, 200, "Taille minimale d'une feuille")],
    "KNN": [("n_neighbors", "int", 1, 100, "Nombre de voisins"),
            ("weights", "choice", ["uniform", "distance"], None, "Pondération des voisins")],
    "Random Forest": [("n_estimators", "int", 50, 500, "Nombre d'arbres"),
                      ("max_depth", "int", 1, 30, "Profondeur maximale"),
                      ("min_samples_leaf", "int", 1, 100, "Taille minimale d'une feuille")],
    "XGBoost": [("n_estimators", "int", 50, 600, "Nombre d'arbres"),
                ("max_depth", "int", 1, 10, "Profondeur maximale"),
                ("learning_rate", "float", 0.005, 0.5, "Taux d'apprentissage"),
                ("subsample", "float", 0.4, 1.0, "Part des lignes par arbre"),
                ("colsample_bytree", "float", 0.4, 1.0, "Part des colonnes par arbre")],
}


def make_model(name, params=None, scale_pos_weight=1.0, seed=42):
    """Construit un classifieur. Déséquilibre : class_weight='balanced' / scale_pos_weight."""
    p = dict(params or {})
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed, **p)
    if name == "Decision Tree":
        return DecisionTreeClassifier(class_weight="balanced", random_state=seed, **p)
    if name == "KNN":
        return KNeighborsClassifier(**p)
    if name == "Random Forest":
        return RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=1, **p)
    if name == "XGBoost":
        return XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=1,
                             scale_pos_weight=scale_pos_weight, **p)
    raise ValueError(f"Modèle inconnu : {name}")


def train_model(name, preprocessor, X_train, y_train, tune=True, params=None, n_iter=10, cv=3, seed=42):
    """
    Entraîne Pipeline(prétraitement -> modèle). Le prétraitement est cloné : pendant la validation croisée
    il est ré-ajusté sur chaque pli (pas de fuite). Renvoie (pipeline, paramètres retenus, AUC CV ou None).
    Variable d'environnement TRAIN_N_JOBS : nombre de processus de la recherche (1 sur un hébergeur à faible mémoire).
    """
    pos = max(int((y_train == 1).sum()), 1)
    spw = (y_train == 0).sum() / pos
    pipe = Pipeline([("pre", clone(preprocessor)), ("model", make_model(name, None, spw, seed))])

    if tune:
        grid = {f"model__{k}": v for k, v in PARAM_GRIDS[name].items()}
        size = math.prod(len(v) for v in grid.values())
        search = RandomizedSearchCV(
            pipe, grid, n_iter=min(n_iter, size), scoring="roc_auc",
            cv=StratifiedKFold(cv, shuffle=True, random_state=seed), n_jobs=int(os.environ.get("TRAIN_N_JOBS", "-1")),
            random_state=seed, refit=True)
        search.fit(X_train, y_train)
        best = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        return search.best_estimator_, best, float(search.best_score_)

    chosen = {**DEFAULT_PARAMS[name], **(params or {})}
    pipe.set_params(**{f"model__{k}": v for k, v in chosen.items()})
    pipe.fit(X_train, y_train)
    return pipe, chosen, None


def predict_proba(pipe, X):
    return pipe.predict_proba(X)[:, 1]


def score_models(pipelines, X):
    """Probabilités de la classe positive pour chaque modèle : {nom: ndarray}."""
    return {name: predict_proba(p, X) for name, p in pipelines.items()}


# --------------------------------------------------------------------------- #
# Métriques (à partir des scores : permet d'évaluer sans avoir les modèles)
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, proba, threshold=0.5):
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, proba)
    fpr, tpr, _ = roc_curve(y_true, proba)
    return {
        "AUC-ROC": auc,
        "PR-AUC": average_precision_score(y_true, proba),
        "Précision": precision_score(y_true, pred, zero_division=0),
        "Rappel": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, pred),
        "Gini": 2 * auc - 1,                 # utile pour le scoring de crédit
        "KS": float(np.max(tpr - fpr)),      # statistique de Kolmogorov-Smirnov
    }


def evaluate_scores(scores, y_true, threshold=0.5):
    """Tableau comparatif (une ligne par modèle), trié par AUC décroissante."""
    rows = {name: compute_metrics(y_true, s, threshold) for name, s in scores.items()}
    return pd.DataFrame(rows).T.sort_values("AUC-ROC", ascending=False).round(4)


def evaluate_models(pipelines, X_test, y_test, threshold=0.5):
    return evaluate_scores(score_models(pipelines, X_test), y_test, threshold)


def threshold_table(y_true, proba, thresholds=None):
    """Précision / rappel / F1 pour une grille de seuils (effet du curseur)."""
    thresholds = np.linspace(0.02, 0.98, 49) if thresholds is None else thresholds
    y_true = np.asarray(y_true)
    rows = []
    for t in thresholds:
        pred = (np.asarray(proba) >= t).astype(int)
        rows.append((t, precision_score(y_true, pred, zero_division=0), recall_score(y_true, pred, zero_division=0),
                     f1_score(y_true, pred, zero_division=0)))
    return pd.DataFrame(rows, columns=["Seuil", "Précision", "Rappel", "F1"])


def best_threshold_f1(y_true, proba):
    t = threshold_table(y_true, proba, np.linspace(0.05, 0.95, 91))
    return float(t.loc[t["F1"].idxmax(), "Seuil"])


def confusion_counts(y_true, proba, threshold=0.5):
    tn, fp, fn, tp = confusion_matrix(np.asarray(y_true), (np.asarray(proba) >= threshold).astype(int),
                                      labels=[0, 1]).ravel()
    return {"VN": int(tn), "FP": int(fp), "FN": int(fn), "VP": int(tp)}
