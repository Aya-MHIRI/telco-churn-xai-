"""
Explicabilité (calculs uniquement, aucun graphique) : importances, SHAP, LIME, comparaison SHAP/LIME,
phrase de synthèse et analyse d'équité. Les graphiques sont dans src/xai_plots.py.

shap et lime sont importés « à la demande » : l'API démarre avec la même empreinte mémoire qu'avant
et ne charge ces bibliothèques que lorsqu'une explication est demandée.
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.preprocess import get_feature_names, transform_to_df

# Libellés français (variables Telco) pour des phrases lisibles ; sinon le nom de colonne est conservé
FR = {
    "tenure": "ancienneté (mois)", "MonthlyCharges": "facture mensuelle", "TotalCharges": "total facturé",
    "Contract": "contrat", "InternetService": "service internet", "OnlineSecurity": "sécurité en ligne",
    "OnlineBackup": "sauvegarde en ligne", "DeviceProtection": "protection de l'appareil",
    "TechSupport": "support technique", "StreamingTV": "TV en streaming", "StreamingMovies": "films en streaming",
    "PaymentMethod": "mode de paiement", "PaperlessBilling": "facture dématérialisée",
    "SeniorCitizen": "senior", "Partner": "en couple", "Dependents": "personnes à charge", "gender": "genre",
    "PhoneService": "service téléphonique", "MultipleLines": "lignes multiples",
    "nb_services": "nombre de services optionnels", "has_protection": "protection (sécurité ou support)",
}


def fr(col):
    return FR.get(col, col)


def n_jobs():
    """Nombre de processus pour les calculs parallèles (TRAIN_N_JOBS=1 sur un hébergeur à faible mémoire)."""
    return int(os.environ.get("TRAIN_N_JOBS", "-1"))


def split_pipeline(pipe):
    return pipe.named_steps["pre"], pipe.named_steps["model"]


def model_kind(model):
    n = type(model).__name__
    if n in ("RandomForestClassifier", "DecisionTreeClassifier", "XGBClassifier"):
        return "tree"
    if n == "LogisticRegression":
        return "linear"
    return "kernel"          # KNN : SHAP par noyau (plus lent)


def shap_units(model):
    return "log-odds" if type(model).__name__ in ("XGBClassifier", "LogisticRegression") else "probabilité"


def origin_of(name, num_cols, cat_cols):
    """Variable d'origine d'une colonne transformée (ex. 'Contract_One year' -> 'Contract')."""
    if name in num_cols:
        return name
    for c in sorted(cat_cols, key=len, reverse=True):
        if name == c or name.startswith(c + "_"):
            return c
    return name


def fmt_value(v):
    if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
        return f"{float(v):.4g}"
    return str(v)


# --------------------------------------------------------------------------- #
# Importances globales
# --------------------------------------------------------------------------- #
def native_importance(pipe):
    """Coefficients (régression logistique) ou importance par impureté (arbres). None pour KNN."""
    pre, model = split_pipeline(pipe)
    names = get_feature_names(pre)
    if hasattr(model, "coef_"):
        imp, kind = np.ravel(model.coef_), "Coefficient (variables standardisées)"
    elif hasattr(model, "feature_importances_"):
        imp, kind = np.ravel(model.feature_importances_), "Importance par impureté"
    else:
        return None
    df = pd.DataFrame({"Variable": names, "Importance": imp})
    df["Type"] = kind
    return df.reindex(df["Importance"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def permutation_importance_df(pipe, X_test, y_test, n_repeats=5, n_max=500, scoring="roc_auc", seed=42):
    """Baisse moyenne de l'AUC quand une variable (d'origine) est mélangée : comparable entre modèles."""
    X, y = X_test, pd.Series(np.asarray(y_test), index=X_test.index)
    if len(X) > n_max:                                   # sous-échantillon : calcul borné en temps
        idx = X.sample(n_max, random_state=seed).index
        X, y = X.loc[idx], y.loc[idx]
    r = permutation_importance(pipe, X, y, scoring=scoring, n_repeats=n_repeats, random_state=seed, n_jobs=n_jobs())
    return (pd.DataFrame({"Variable": X.columns, "Importance": r.importances_mean, "Écart-type": r.importances_std})
            .sort_values("Importance", ascending=False).reset_index(drop=True))


def permutation_all(pipelines, X_test, y_test, n_repeats=5, n_max=500, progress=None):
    frames = []
    for i, (name, p) in enumerate(pipelines.items()):
        if progress:
            progress(i, len(pipelines), f"Permutation : {name}…")
        d = permutation_importance_df(p, X_test, y_test, n_repeats, n_max)
        d["Modèle"] = name
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #
@dataclass
class ShapResult:
    values: np.ndarray          # (n, p) contributions de chaque colonne transformée
    base: np.ndarray            # (n,) valeur de base (sortie moyenne du modèle)
    data: pd.DataFrame          # (n, p) colonnes transformées (échelle du modèle)
    feature_names: list
    units: str


def _explainer(pipe, name, background_raw, cache):
    key = ("shap", name)
    if key not in cache:
        import shap
        pre, model = split_pipeline(pipe)
        bg = transform_to_df(pre, background_raw).values
        kind = model_kind(model)
        if kind == "tree":
            ex = shap.TreeExplainer(model)
        elif kind == "linear":
            ex = shap.LinearExplainer(model, bg)
        else:
            ex = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], shap.kmeans(bg, 10))
        cache[key] = (ex, kind)
    return cache[key]


def shap_values(pipe, name, X_raw, background_raw, cache, kernel_nsamples=150):
    """Valeurs SHAP de la classe positive pour les lignes de X_raw."""
    pre, model = split_pipeline(pipe)
    Xt = transform_to_df(pre, X_raw)
    ex, kind = _explainer(pipe, name, background_raw, cache)
    if kind == "kernel":
        vals = np.asarray(ex.shap_values(Xt.values, nsamples=kernel_nsamples, silent=True))
        base = np.full(len(Xt), float(np.ravel(ex.expected_value)[0]))
    else:
        exp = ex(Xt.values, check_additivity=False) if kind == "tree" else ex(Xt.values)
        vals, base = np.asarray(exp.values), np.asarray(exp.base_values)
        if vals.ndim == 3:                       # RandomForest / DecisionTree : une sortie par classe
            vals = vals[:, :, 1]
            base = base[:, 1] if base.ndim == 2 else base
    base = np.broadcast_to(np.ravel(base), (len(Xt),)).copy()
    return ShapResult(vals, base, Xt, list(Xt.columns), shap_units(model))


def unscaled_df(pre, Xt_df, num_cols):
    """Remet les variables numériques à leur échelle d'origine (pour des graphiques lisibles)."""
    out = Xt_df.copy()
    tr = pre.named_transformers_.get("num")
    scaler = tr.named_steps.get("scaler") if tr is not None else None
    cols = [c for c in num_cols if c in out.columns]
    if scaler is not None and cols:
        out[cols] = scaler.inverse_transform(out[cols].values)
    return out


def shap_importance_by_feature(values, feature_names, num_cols, cat_cols):
    """Importance SHAP moyenne (|SHAP|) agrégée par variable d'origine (les colonnes one-hot sont additionnées)."""
    origins = [origin_of(n, num_cols, cat_cols) for n in feature_names]
    df = pd.DataFrame(np.asarray(values), columns=origins)
    agg = df.T.groupby(level=0).sum().T
    imp = agg.abs().mean().sort_values(ascending=False)
    out = pd.DataFrame({"Variable": imp.index, "Importance SHAP moyenne": imp.values})
    out["Part"] = out["Importance SHAP moyenne"] / out["Importance SHAP moyenne"].sum()
    return out.reset_index(drop=True)


def contributions_df(values_row, feature_names, raw_row, num_cols, cat_cols):
    """Contribution de chaque variable d'origine pour un individu (colonnes one-hot additionnées)."""
    df = pd.DataFrame({"col": feature_names, "contribution": np.asarray(values_row)})
    df["feature"] = [origin_of(n, num_cols, cat_cols) for n in df["col"]]
    g = df.groupby("feature", sort=False)["contribution"].sum().reset_index()
    g["valeur"] = [raw_row.get(f, np.nan) for f in g["feature"]]
    g["label"] = [f"{fr(f)} = {fmt_value(v)}" for f, v in zip(g["feature"], g["valeur"])]
    return g.reindex(g["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# LIME
# --------------------------------------------------------------------------- #
class LimeContext:
    """LIME travaille dans l'échelle d'origine des variables numériques (règles lisibles)."""

    def __init__(self, pipe, train_raw, num_cols, seed=42):
        from lime.lime_tabular import LimeTabularExplainer
        self.pre, self.model = split_pipeline(pipe)
        Xt = transform_to_df(self.pre, train_raw)
        self.names = list(Xt.columns)
        self.num_idx = [self.names.index(c) for c in num_cols if c in self.names]
        tr = self.pre.named_transformers_.get("num")
        self.scaler = tr.named_steps.get("scaler") if tr is not None else None
        data = Xt.values.astype(float).copy()
        if self.scaler is not None and self.num_idx:
            data[:, self.num_idx] = self.scaler.inverse_transform(data[:, self.num_idx])
        cat_idx = [i for i in range(len(self.names)) if i not in self.num_idx]
        self.explainer = LimeTabularExplainer(
            data, feature_names=self.names, categorical_features=cat_idx, class_names=["Négatif", "Positif"],
            discretize_continuous=True, mode="classification", random_state=seed)

    def predict_fn(self, arr):
        a = np.array(arr, dtype=float, copy=True)
        if self.scaler is not None and self.num_idx:
            a[:, self.num_idx] = self.scaler.transform(a[:, self.num_idx])
        return self.model.predict_proba(a)

    def instance(self, X_row_raw):
        x = transform_to_df(self.pre, X_row_raw).values.astype(float).copy()
        if self.scaler is not None and self.num_idx:
            x[:, self.num_idx] = self.scaler.inverse_transform(x[:, self.num_idx])
        return x[0]


def lime_context(pipe, name, train_raw, num_cols, cache):
    key = ("lime", name)
    if key not in cache:
        cache[key] = LimeContext(pipe, train_raw, num_cols)
    return cache[key]


def lime_explain(ctx, X_row_raw, num_cols, cat_cols, num_features=10, num_samples=2000):
    """Explication LIME d'un individu (classe positive) : règles, poids et qualité de l'approximation locale."""
    exp = ctx.explainer.explain_instance(ctx.instance(X_row_raw), ctx.predict_fn, num_features=num_features,
                                         num_samples=num_samples, labels=(1,))
    rules = exp.as_list(label=1)
    idx = [i for i, _ in exp.as_map()[1]]
    rows = [{"feature": origin_of(ctx.names[i], num_cols, cat_cols), "rule": r, "weight": float(w)}
            for i, (r, w) in zip(idx, rules)]
    return {"table": pd.DataFrame(rows), "score": float(exp.score), "local_pred": float(np.ravel(exp.local_pred)[0])}


def compare_shap_lime(contrib, lime_table, k=5):
    """Cohérence SHAP / LIME au niveau des variables d'origine : recouvrement du top-k et accord des signes."""
    lime_agg = lime_table.groupby("feature", sort=False)["weight"].sum()
    shap_s = contrib.set_index("feature")["contribution"]
    shap_top = list(shap_s.reindex(shap_s.abs().sort_values(ascending=False).index).head(k).index)
    lime_top = list(lime_agg.reindex(lime_agg.abs().sort_values(ascending=False).index).head(k).index)
    union = list(dict.fromkeys(shap_top + lime_top))
    labels = contrib.set_index("feature")["label"]
    table = pd.DataFrame({
        "feature": union,
        "label": [labels.get(f, f) for f in union],
        "SHAP": [shap_s.get(f, np.nan) for f in union],
        "LIME": [lime_agg.get(f, np.nan) for f in union],
        "Rang SHAP": [shap_top.index(f) + 1 if f in shap_top else np.nan for f in union],
        "Rang LIME": [lime_top.index(f) + 1 if f in lime_top else np.nan for f in union],
    })
    both = table.dropna(subset=["SHAP", "LIME"])
    table["Même sens"] = [None if (pd.isna(a) or pd.isna(b)) else bool(np.sign(a) == np.sign(b))
                          for a, b in zip(table["SHAP"], table["LIME"])]
    common = set(shap_top) & set(lime_top)
    agree = float((np.sign(both["SHAP"]) == np.sign(both["LIME"])).mean()) if len(both) else float("nan")
    return {"k": k, "overlap": len(common), "overlap_pct": len(common) / k, "common": sorted(common),
            "sign_agreement": agree, "table": table}


# --------------------------------------------------------------------------- #
# Phrase de synthèse
# --------------------------------------------------------------------------- #
def _join(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " et " + items[-1]


def generate_summary(contrib, proba, threshold, positive_name="départ", top_k=3):
    """Phrase en langage naturel à partir des plus fortes contributions SHAP."""
    keep = contrib[contrib["contribution"].abs() >= 0.05 * contrib["contribution"].abs().sum()]
    up = keep[keep["contribution"] > 0].sort_values("contribution", ascending=False).head(top_k)
    high = proba >= threshold
    down = keep[keep["contribution"] < 0].sort_values("contribution").head(2 if high else top_k)
    s = f"Le risque de {positive_name} est {'élevé' if high else 'faible'} ({proba:.0%}, seuil de décision {threshold:.0%})"
    if high:
        s += (f", principalement à cause de : {_join(up['label'])}." if len(up) else ".")
        if len(down):
            s += f" Il est atténué par : {_join(down['label'])}."
    else:
        s += (f", principalement grâce à : {_join(down['label'])}." if len(down) else ".")
        if len(up):
            s += f" Les principaux facteurs de risque restants : {_join(up.head(2)['label'])}."
    return s


# --------------------------------------------------------------------------- #
# Équité (variables sensibles)
# --------------------------------------------------------------------------- #
def fairness_report(pipe, X_test, y_test, sensitive_cols, threshold=0.5, bins=4):
    """Taux de prédictions positives, rappel et faux positifs par groupe d'une variable sensible."""
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    y = np.asarray(y_test)
    rows = []
    for col in sensitive_cols:
        if col not in X_test.columns:
            continue
        s = X_test[col]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
            s = pd.qcut(s, q=bins, duplicates="drop")
        s = s.astype(str)
        for g in pd.unique(s):
            m = (s == g).values
            pos, neg = (y[m] == 1), (y[m] == 0)
            rows.append({
                "Variable": col, "Groupe": g, "Effectif": int(m.sum()),
                "Taux réel positif": float(y[m].mean()), "Taux prédit positif": float(pred[m].mean()),
                "Proba moyenne": float(proba[m].mean()),
                "Rappel": float(pred[m][pos].mean()) if pos.any() else np.nan,
                "Faux positifs (FPR)": float(pred[m][neg].mean()) if neg.any() else np.nan})
    return pd.DataFrame(rows)


def parity_gaps(report):
    """Écart max entre groupes du taux de prédictions positives (parité démographique)."""
    if report.empty:
        return pd.DataFrame(columns=["Variable", "Écart de parité"])
    g = report.groupby("Variable")["Taux prédit positif"].agg(lambda s: s.max() - s.min())
    return g.rename("Écart de parité").reset_index()
