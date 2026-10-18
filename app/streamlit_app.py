"""
Application web interactive de classification (Streamlit).

Lancement :  streamlit run app/streamlit_app.py
Variable d'environnement (ou secret Streamlit) : API_URL = adresse de l'API FastAPI (défaut : http://localhost:8000)
"""
import io
import json
import os
import sys
import time
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import streamlit as st

from src import data as D
from src import eda
from src import models as M
from src import explain_service as EX
from src import plots
from src import preprocess as P
from src import trainer as T
from src import xai_plots as XP
from src.api_client import api_get, api_health, api_post, get_api_url

st.set_page_config(page_title="Classification interactive", page_icon="📊", layout="wide")
st.session_state.setdefault("threshold", 0.5)

IMPUTERS = {"Médiane": "median", "Moyenne": "mean", "Zéro (0)": "zero"}
SENSITIVE_HINTS = ["gender", "Gender", "sex", "Sex", "SeniorCitizen", "Partner", "Dependents", "age", "Age",
                   "race", "Race", "marital", "Marital", "Married"]


class TrainingError(Exception):
    pass


def show_table(df, fmt=None, highlight_max=False):
    """Affiche un tableau (avec mise en forme si possible)."""
    try:
        sty = df.style
        if fmt:
            sty = sty.format(fmt, na_rep="—")
        if highlight_max:
            sty = sty.highlight_max(axis=0, color="#cfe8cf")
        st.dataframe(sty, width="stretch")
    except Exception:
        st.dataframe(df, width="stretch")


# --------------------------------------------------------------------------- #
# Module 5 : outils communs (appel de l'API ou calcul local selon la source de l'entraînement)
# --------------------------------------------------------------------------- #
def is_api(tr):
    return tr["source"] == "API FastAPI"


def cached(key, fn):
    """Garde le résultat d'un calcul d'explication dans la session Streamlit (évite de le refaire à chaque clic)."""
    xs = st.session_state.setdefault("xai", {})
    if key not in xs:
        xs[key] = fn()
    return xs[key]


def xai_sync(tr, method, path, local_fn, payload=None, params=None, timeout=180):
    """Calcul court : appel direct de l'API (session entraînée côté API) ou calcul local."""
    if is_api(tr):
        full = f"/sessions/{tr['session_id']}{path}"
        if method == "POST":
            return api_post(full, payload, timeout=timeout)
        return api_get(full, params=params, timeout=timeout)
    return local_fn(st.session_state["local_session"])


def xai_job(tr, path, payload, local_fn, timeout_s=900):
    """Calcul long : tâche de fond côté API (avec avancement) ou calcul local avec barre de progression."""
    bar, status = st.progress(0.0), st.empty()
    try:
        if is_api(tr):
            job = api_post(f"/sessions/{tr['session_id']}{path}", payload, timeout=60)
            t0 = time.time()
            while True:
                j = api_get(f"/jobs/{job['job_id']}")
                bar.progress(min(float(j["progress"]), 1.0))
                status.write(f"⚙️ {j['message']}")
                if j["status"] == "done":
                    return j["result"]
                if j["status"] == "error":
                    raise TrainingError(j["error"])
                if time.time() - t0 > timeout_s:
                    raise TrainingError("Délai dépassé.")
                time.sleep(1.0)

        def cb(i, n, msg):
            bar.progress(min(i / n, 1.0) if n else 0.0)
            status.write(f"⚙️ {msg}")
        return local_fn(st.session_state["local_session"], cb)
    finally:
        status.empty()
        bar.empty()


def show_xai_error(e):
    if isinstance(e, requests.RequestException):
        st.error(f"API injoignable ({type(e).__name__}). Elle est peut-être en veille : réessayez dans une minute.")
    elif "Session introuvable" in str(e):
        st.error("La session d'entraînement a expiré côté API (redémarrage du service). "
                 "Ré-entraînez les modèles dans l'onglet 3.")
    else:
        st.error(f"{e}")


def render_form(schema, form_key, button):
    """Formulaire de saisie d'un individu, construit à partir du schéma des variables. -> (valeurs, soumis)."""
    with st.form(form_key):
        fcols = st.columns(3)
        data_in = {}
        for i, (c, sc) in enumerate(schema.items()):
            with fcols[i % 3]:
                if sc["type"] == "cat":
                    ch = sc["choices"]
                    data_in[c] = st.selectbox(c, ch, index=ch.index(sc["mode"]), key=f"{form_key}_{c}")
                elif float(sc["min"]).is_integer() and float(sc["max"]).is_integer():
                    data_in[c] = int(st.number_input(c, int(sc["min"]), int(sc["max"]), int(sc["median"]),
                                                     key=f"{form_key}_{c}"))
                else:
                    data_in[c] = float(st.number_input(c, float(sc["min"]), float(sc["max"]), float(sc["median"]),
                                                       key=f"{form_key}_{c}"))
        submitted = st.form_submit_button(button, type="primary")
    return data_in, submitted


# =========================================================================== #
# Barre latérale : source des données, cible, API
# =========================================================================== #
st.sidebar.title("📊 Classification interactive")
st.sidebar.caption("Data Mining · ENSI · Cas 13 : Telco Churn")
source = st.sidebar.radio("Source des données", ["Telco Churn (téléchargement automatique)", "Importer un fichier CSV"])
IS_TELCO_SOURCE = source.startswith("Telco")


@st.cache_data(show_spinner="Téléchargement automatique du jeu de données…")
def load_default():
    return D.load_data()


@st.cache_data(show_spinner="Lecture du fichier…")
def load_csv(content: bytes):
    return pd.read_csv(io.BytesIO(content), sep=None, engine="python")


upload_name = None
if IS_TELCO_SOURCE:
    try:
        df_raw = load_default()
    except Exception as e:
        st.error(f"Impossible de charger le jeu de données : {e}")
        st.stop()
else:
    up = st.sidebar.file_uploader("Fichier CSV", type=["csv"])
    if up is None:
        st.title("📊 Classification interactive")
        st.info("Importez un fichier CSV dans la barre latérale, puis indiquez la variable cible.")
        st.stop()
    upload_name = up.name
    df_raw = load_csv(up.getvalue())

df_clean = D.generic_clean(df_raw)
cols = list(df_clean.columns)
default_target = "Churn" if "Churn" in cols else cols[-1]
target = st.sidebar.selectbox("Variable cible", cols, index=cols.index(default_target))

values = df_clean[target].dropna()
if set(values.unique()) <= {0, 1}:
    positive = 1
    st.sidebar.caption("Cible binaire 0/1 : la classe positive est 1.")
else:
    if pd.api.types.is_numeric_dtype(values) and values.nunique() > 20:
        st.sidebar.error("Cible continue : choisissez une cible catégorielle ou binaire.")
        st.stop()
    options = list(values.value_counts().index)
    idx = options.index("Yes") if "Yes" in options else len(options) - 1
    positive = st.sidebar.selectbox("Classe positive (à détecter)", options, index=idx)
    if len(options) > 2:
        st.sidebar.caption("Plus de 2 classes : ramené à « classe positive » contre toutes les autres.")

df = df_clean.dropna(subset=[target]).copy()
df[target] = P.encode_target(df[target], positive).astype(int)
FEATURES_ALL = [c for c in df.columns if c != target]
NUM_ALL = [c for c in FEATURES_ALL if pd.api.types.is_numeric_dtype(df[c])]

# Réinitialiser les résultats si le jeu de données ou la cible change
sig = (source, upload_name, target, str(positive))
if st.session_state.get("sig") != sig:
    for k in ["prep", "train", "local_session", "last_pred", "xai"]:
        st.session_state.pop(k, None)
    st.session_state["sig"] = sig

st.sidebar.divider()
st.sidebar.markdown("**API FastAPI**")
st.sidebar.caption(f"`{get_api_url()}`")
if st.sidebar.button("Tester la connexion"):
    ok, msg = api_health()
    (st.sidebar.success if ok else st.sidebar.error)(f"{'Disponible' if ok else 'Injoignable'} ({msg})")

st.title("📊 Application de classification interactive")
st.caption(f"Cible : **{target}** (classe positive : {positive}) · {len(df):,} observations · "
           f"{len(FEATURES_ALL)} variables explicatives".replace(",", " "))

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["1️⃣ Données", "2️⃣ Préparation", "3️⃣ Modélisation", "4️⃣ Évaluation",
                                              "5️⃣ Explicabilité (XAI)", "6️⃣ Prédiction"])

# =========================================================================== #
# MODULE 1 : Données
# =========================================================================== #
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", f"{len(df_raw):,}".replace(",", " "))
    c2.metric("Colonnes", df_raw.shape[1])
    c3.metric("Classe positive", f"{df[target].mean():.1%}")
    miss_tab = eda.missing_values_table(df_raw)
    c4.metric("Valeurs manquantes", int(miss_tab["Total manquants"].sum()))

    st.subheader("Aperçu des données")
    st.dataframe(df_raw.head(20), width="stretch")
    with st.expander("Statistiques descriptives"):
        st.markdown("**Variables numériques**")
        st.dataframe(df_clean.select_dtypes("number").describe().T.round(2), width="stretch")
        obj = df_clean.select_dtypes(exclude="number")
        if not obj.empty:
            st.markdown("**Variables catégorielles**")
            st.dataframe(obj.describe().T, width="stretch")

    st.subheader("Valeurs manquantes")
    miss = miss_tab[miss_tab["Total manquants"] > 0]
    if miss.empty:
        st.success("Aucune valeur manquante détectée.")
    else:
        st.dataframe(miss, width="stretch")
        st.caption("« Vides cachés » = cases contenant seulement des espaces (ex. `TotalCharges` pour les clients "
                   "sans ancienneté). Elles sont converties en valeurs manquantes puis imputées.")

    st.subheader("Proportion de chaque classe")
    cl1, cl2 = st.columns([2, 1])
    cl1.plotly_chart(eda.plot_class_balance(df, target), width="stretch")
    cl2.dataframe(eda.class_balance(df, target), width="stretch", hide_index=True)
    if df[target].mean() < 0.35 or df[target].mean() > 0.65:
        cl2.warning("Classes déséquilibrées : on privilégie AUC, PR-AUC, rappel et F1 plutôt que l'accuracy.")

    st.subheader("Visualisations exploratoires interactives")
    v1, v2, v3, v4 = st.tabs(["Distributions", "Corrélations", "Taux de la classe positive", "Croisements"])
    with v1:
        if NUM_ALL:
            var = st.selectbox("Variable numérique", NUM_ALL, key="dist_var")
            st.plotly_chart(eda.plot_numeric_distribution(df, var, target), width="stretch")
            st.plotly_chart(eda.plot_box_by_target(df, var, target), width="stretch")
        else:
            st.info("Aucune variable numérique.")
    with v2:
        if NUM_ALL:
            st.plotly_chart(eda.plot_correlation(df, target, NUM_ALL), width="stretch")
            thr_c = st.slider("Seuil de corrélation (redondance)", 0.5, 0.95, 0.7, 0.05)
            hc = eda.high_correlations(df, NUM_ALL, thr_c)
            if hc.empty:
                st.success("Aucune paire de variables au-dessus du seuil.")
            else:
                st.warning("Variables redondantes :")
                st.dataframe(hc, width="stretch", hide_index=True)
        else:
            st.info("Aucune variable numérique.")
    with v3:
        var = st.selectbox("Variable", FEATURES_ALL, key="rate_var")
        st.plotly_chart(eda.plot_rate_by_category(df, var, target), width="stretch")
        with st.expander("Classement de toutes les variables par pouvoir discriminant"):
            st.plotly_chart(eda.plot_rate_spread(df, target, FEATURES_ALL), width="stretch")
    with v4:
        if len(FEATURES_ALL) >= 2:
            ra = "Contract" if "Contract" in FEATURES_ALL else FEATURES_ALL[0]
            rb = "InternetService" if "InternetService" in FEATURES_ALL else FEATURES_ALL[1]
            cc1, cc2 = st.columns(2)
            row = cc1.selectbox("Lignes", FEATURES_ALL, index=FEATURES_ALL.index(ra))
            col = cc2.selectbox("Colonnes", FEATURES_ALL, index=FEATURES_ALL.index(rb))
            st.plotly_chart(eda.plot_crosstab_rate(df, row, col, target), width="stretch")

# =========================================================================== #
# MODULE 2 : Préparation
# =========================================================================== #
with tab2:
    st.subheader("Variables")
    st.caption(f"Cible choisie dans la barre latérale : **{target}**.")
    features = st.multiselect("Variables explicatives", FEATURES_ALL, default=FEATURES_ALL)

    st.subheader("Traitements")
    p1, p2, p3 = st.columns(3)
    imputer_lbl = p1.selectbox("Valeurs manquantes numériques", list(IMPUTERS), index=0,
                               help="Les manquantes catégorielles sont remplacées par la modalité la plus fréquente.")
    scale = p2.checkbox("Normaliser (StandardScaler)", value=True,
                        help="Indispensable pour la régression logistique et KNN.")
    test_pct = p3.slider("Taille du jeu de test (%)", 10, 40, 20, 5)
    q1, q2, q3 = st.columns(3)
    simplify = q1.checkbox("Regrouper « No internet/phone service » en « No »", value=True)
    engineer = q2.checkbox("Créer des variables simples (nb_services, has_protection)", value=True,
                           help="Actif seulement si les colonnes de services existent (jeu Telco).")
    seed = q3.number_input("Graine aléatoire", 0, 9999, 42)
    st.caption("Encodage : OneHotEncoder sur les variables catégorielles. Séparation train/test **stratifiée**. "
               "Le transformateur est ajusté sur le jeu d'entraînement uniquement (pas de fuite de données).")

    if st.button("✅ Appliquer la préparation", type="primary"):
        if not features:
            st.error("Sélectionnez au moins une variable explicative.")
        else:
            try:
                st.session_state["prep"] = P.prepare(
                    df, target, features, test_size=test_pct / 100, imputer=IMPUTERS[imputer_lbl], scale=scale,
                    simplify=simplify, engineer=engineer, seed=int(seed))
                for k in ["train", "local_session", "last_pred", "xai"]:
                    st.session_state.pop(k, None)
            except Exception as e:
                st.error(f"Erreur de préparation : {e}")

    prep = st.session_state.get("prep")
    if prep:
        st.success("Préparation appliquée.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Entraînement", len(prep["X_train"]))
        m2.metric("Test", len(prep["X_test"]))
        m3.metric("Classe positive (train)", f"{prep['y_train'].mean():.1%}")
        m4.metric("Classe positive (test)", f"{prep['y_test'].mean():.1%}")
        st.write(f"**Numériques ({len(prep['num_cols'])})** : {', '.join(prep['num_cols']) or '—'}")
        st.write(f"**Catégorielles ({len(prep['cat_cols'])})** : {', '.join(prep['cat_cols']) or '—'}")
        st.write(f"**Colonnes après encodage : {len(prep['feature_names'])}**")
        with st.expander("Aperçu des données transformées (entraînement)"):
            st.dataframe(P.transform_to_df(prep["preprocessor"], prep["X_train"].head(10)).round(3), width="stretch")
        st.caption("Ces réglages sont envoyés tels quels à l'API au moment de l'entraînement (onglet 3).")
    else:
        st.info("Cliquez sur « Appliquer la préparation » pour continuer.")

# =========================================================================== #
# MODULE 3 : Modélisation (entraînement via l'API FastAPI)
# =========================================================================== #
with tab3:
    prep = st.session_state.get("prep")
    if not prep:
        st.info("Appliquez d'abord la préparation (onglet 2).")
    else:
        chosen = st.multiselect("Modèles à entraîner", M.MODEL_NAMES, default=M.MODEL_NAMES)
        tune = st.checkbox("Optimiser les hyperparamètres (RandomizedSearchCV, score AUC)", value=True)
        user_params, n_iter, cv = {}, 10, 3
        if tune:
            t1, t2 = st.columns(2)
            n_iter = t1.slider("Combinaisons testées par modèle", 3, 30, 10)
            cv = t2.slider("Plis de validation croisée", 2, 5, 3)
            with st.expander("Grilles de recherche utilisées"):
                for n in chosen:
                    st.write(f"**{n}** : " + " · ".join(f"`{k}` {v}" for k, v in M.PARAM_GRIDS[n].items()))
        else:
            st.caption("Réglez les hyperparamètres de chaque modèle :")
            for n in chosen:
                with st.expander(f"Hyperparamètres : {n}"):
                    prm = {}
                    for pname, kind, lo, hi, helptxt in M.PARAM_SPECS[n]:
                        default = M.DEFAULT_PARAMS[n][pname]
                        label, key = f"{pname} : {helptxt}", f"hp_{n}_{pname}"
                        if kind == "int":
                            prm[pname] = int(st.number_input(label, int(lo), int(hi), int(default), key=key))
                        elif kind == "float":
                            prm[pname] = float(st.number_input(label, float(lo), float(hi), float(default),
                                                               format="%.3f", key=key))
                        else:
                            prm[pname] = st.selectbox(label, lo, index=lo.index(default), key=key)
                    user_params[n] = prm

        use_api = st.toggle("Entraîner via l'API FastAPI", value=True,
                            help="Désactivé : entraînement local dans l'application. Si l'API est injoignable, "
                                 "l'application bascule automatiquement en local.")

        if st.button("🚀 Entraîner les modèles", type="primary", disabled=not chosen):
            s = prep["settings"]
            cfg = {"target": target, "positive_label": 1, "features": prep["raw_features"],
                   "test_size": s["test_size"], "imputer": s["imputer"], "scale": s["scale"],
                   "simplify": s["simplify"], "engineer": s["engineer"], "seed": s["seed"],
                   "models": chosen, "tune": tune, "n_iter": n_iter, "cv": cv, "params": user_params}
            df_train = df[prep["raw_features"] + [target]]
            bar, status = st.progress(0.0), st.empty()
            result, where = None, "local"
            try:
                if use_api:
                    try:
                        records = json.loads(df_train.to_json(orient="records"))
                        job = api_post("/train", {"data": records, **cfg}, timeout=180)
                        t0 = time.time()
                        while result is None:
                            j = api_get(f"/jobs/{job['job_id']}")
                            bar.progress(min(float(j["progress"]), 1.0))
                            status.write(f"⚙️ {j['message']}")
                            if j["status"] == "done":
                                result, where = j["result"], "API FastAPI"
                            elif j["status"] == "error":
                                raise TrainingError(j["error"])
                            elif time.time() - t0 > 1800:
                                raise TrainingError("Délai dépassé (30 min).")
                            else:
                                time.sleep(1.0)
                    except requests.RequestException as e:
                        st.warning(f"API injoignable ({type(e).__name__}) : entraînement local.")
                    except RuntimeError as e:      # erreur renvoyée par l'API (validation, session...)
                        raise TrainingError(str(e))
                if result is None:
                    def cb(i, n, msg):
                        bar.progress(min(i / n, 1.0))
                        status.write(f"⚙️ {msg}")
                    session = T.run_training(cfg, df_train, progress=cb)
                    st.session_state["local_session"] = session
                    result, where = T.session_summary(session), "local"
                status.empty()
                bar.progress(1.0)
                st.session_state["train"] = {
                    "source": where, "session_id": result.get("session_id"), "models": result["models"],
                    "tid": int(time.time() * 1000),
                    "scores": {m: np.array(v) for m, v in result["scores"].items()},
                    "y_test": np.array(result["y_test"]), "best_params": result["best_params"],
                    "cv_scores": result["cv_scores"], "n_train": result["n_train"], "n_test": result["n_test"],
                    "schema": result["schema"], "raw_features": result["raw_features"]}
                st.session_state.pop("last_pred", None)
                st.session_state.pop("xai", None)
            except TrainingError as e:
                status.empty()
                st.error(f"Échec de l'entraînement : {e}")
            except Exception as e:
                status.empty()
                st.error(f"Erreur : {type(e).__name__}: {e}")

        tr = st.session_state.get("train")
        if tr:
            st.success(f"Modèles entraînés ({tr['source']}) : {', '.join(tr['models'])} · "
                       f"{tr['n_train']} lignes d'entraînement, {tr['n_test']} de test.")
            st.subheader("Tableau comparatif des modèles (jeu de test)")
            res = M.evaluate_scores(tr["scores"], tr["y_test"], st.session_state["threshold"])
            if any(v is not None for v in tr["cv_scores"].values()):
                res["AUC (validation croisée)"] = pd.Series(tr["cv_scores"])
            st.caption(f"Métriques au seuil de décision {st.session_state['threshold']:.2f} (modifiable dans l'onglet 4).")
            show_table(res, fmt="{:.3f}", highlight_max=True)
            with st.expander("Hyperparamètres retenus"):
                st.dataframe(pd.DataFrame({k: {a: str(b) for a, b in v.items()}
                                           for k, v in tr["best_params"].items()}).T.fillna("—"), width="stretch")

# =========================================================================== #
# MODULE 4 : Évaluation
# =========================================================================== #
with tab4:
    tr = st.session_state.get("train")
    if not tr:
        st.info("Entraînez d'abord des modèles (onglet 3).")
    else:
        scores, yte = tr["scores"], tr["y_test"]
        res = M.evaluate_scores(scores, yte, st.session_state["threshold"])
        best = res.index[0]
        st.subheader("Métriques de classification")
        st.caption("Rappel, précision et F1 sont calculés sur la classe positive. Gini = 2·AUC−1 et KS : utiles "
                   "pour les cas de scoring de crédit.")
        show_table(res, fmt="{:.3f}", highlight_max=True)

        e1, e2 = st.columns(2)
        e1.plotly_chart(plots.plot_roc(scores, yte), width="stretch")
        e2.plotly_chart(plots.plot_pr(scores, yte), width="stretch")

        st.subheader("Seuil de décision et matrice de confusion")
        sel = st.selectbox("Modèle analysé", list(scores), index=list(scores).index(best), key="eval_model")
        proba = scores[sel]

        def use_best_threshold():
            t = st.session_state["train"]
            st.session_state["threshold"] = round(
                M.best_threshold_f1(t["y_test"], t["scores"][st.session_state["eval_model"]]), 2)

        s1, s2 = st.columns([3, 1])
        thr = s1.slider("Seuil de décision (probabilité à partir de laquelle on prédit la classe positive)",
                        0.01, 0.99, step=0.01, key="threshold")
        s2.write("")
        s2.button("🎯 Seuil optimal (F1)", on_click=use_best_threshold, width="stretch")

        mt = M.compute_metrics(yte, proba, thr)
        cc = M.confusion_counts(yte, proba, thr)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Précision", f"{mt['Précision']:.1%}")
        k2.metric("Rappel", f"{mt['Rappel']:.1%}")
        k3.metric("F1", f"{mt['F1']:.3f}")
        k4.metric("Positifs ratés (FN)", cc["FN"])
        st.caption("Baisser le seuil : on détecte plus de positifs (rappel ↑) mais avec plus de fausses alertes "
                   "(précision ↓). Monter le seuil fait l'inverse.")
        g1, g2 = st.columns(2)
        g1.plotly_chart(plots.plot_confusion(yte, proba, thr, title=f"Matrice de confusion : {sel}"), width="stretch")
        g2.plotly_chart(plots.plot_threshold_curve(yte, proba, thr), width="stretch")

        st.subheader("Compromis performance / interprétabilité")
        if "Logistic Regression" in res.index and best != "Logistic Regression":
            gap = res.loc[best, "AUC-ROC"] - res.loc["Logistic Regression", "AUC-ROC"]
            st.info(f"Meilleur modèle : **{best}** (AUC {res.loc[best, 'AUC-ROC']:.3f}). Écart d'AUC avec la régression "
                    f"logistique, plus simple à interpréter : **{gap:+.3f}**. Si cet écart est faible, le modèle "
                    "simple est un choix défendable pour un décideur.")
        elif best == "Logistic Regression":
            st.info("La régression logistique est aussi le meilleur modèle : performance et simplicité se rejoignent.")

# =========================================================================== #
# MODULE 5 : Explicabilité (XAI)
# =========================================================================== #
with tab5:
    tr = st.session_state.get("train")
    if not tr:
        st.info("Entraînez d'abord des modèles (onglet 3).")
    else:
        thr = st.session_state["threshold"]
        tid = tr["tid"]
        xs = st.session_state.setdefault("xai", {})
        order = list(M.evaluate_scores(tr["scores"], tr["y_test"], thr).index)
        mname = st.selectbox("Modèle à expliquer", order, key="xai_model",
                             help="Classés du meilleur au moins bon (AUC sur le jeu de test).")
        st.caption(f"Calculs effectués par : **{tr['source']}**" + (
            "" if not is_api(tr) else " (l'instance gratuite peut mettre quelques secondes à répondre)."))
        xg, xl, xe = st.tabs(["🌍 Explications globales", "👤 Explication locale", "⚖️ Équité"])

        # ------------------------------ Globales -------------------------------- #
        with xg:
            st.subheader("Importance propre au modèle")
            try:
                nat = cached(("native", tid, mname), lambda: xai_sync(
                    tr, "GET", "/explain/native", lambda s_: EX.native(s_, mname), params={"model": mname}))
                if nat["available"]:
                    st.plotly_chart(XP.plot_native_importance(nat["importance"]), width="stretch")
                else:
                    st.info("KNN n'a pas d'importance interne : utilisez l'importance par permutation ci-dessous.")
            except Exception as e:
                show_xai_error(e)

            st.subheader("Importance par permutation (comparable entre modèles)")
            st.caption("Baisse d'AUC quand les valeurs d'une variable sont mélangées au hasard : plus la baisse est "
                       "grande, plus le modèle dépend de la variable. Calculée pour tous les modèles entraînés.")
            if st.button("Calculer l'importance par permutation"):
                try:
                    res = xai_job(tr, "/explain/permutation", {"n_repeats": 5, "n_max": 500},
                                  lambda s_, cb: EX.permutation(s_, 5, 500, cb))
                    xs[("perm", tid)] = res["importance"]
                except Exception as e:
                    show_xai_error(e)
            if ("perm", tid) in xs:
                st.plotly_chart(XP.plot_permutation_comparison(xs[("perm", tid)]), width="stretch")

            st.subheader("SHAP : effet des variables sur les prédictions")
            is_knn = mname == "KNN"
            n_max = 60 if is_knn else 500
            n_s = st.slider("Nombre d'individus du jeu de test utilisés", 20, min(n_max, tr["n_test"]),
                            min(60 if is_knn else 200, tr["n_test"]), key="shap_n",
                            help="Plus il y en a, plus le graphique est fiable, mais plus le calcul est long.")
            if is_knn:
                st.caption("KNN : SHAP par noyau (KernelExplainer), plus lent, donc échantillon limité à 60.")
            if st.button("Calculer SHAP"):
                try:
                    xs[("shapg", tid, mname)] = xai_job(
                        tr, "/explain/shap-global", {"model": mname, "n": int(n_s)},
                        lambda s_, cb: EX.shap_global(s_, mname, int(n_s), cb))
                except Exception as e:
                    show_xai_error(e)
            sg = xs.get(("shapg", tid, mname))
            if sg:
                st.caption(f"Calculé sur {sg['n']} individus du jeu de test. Unités : {sg['units']}.")
                st.plotly_chart(XP.plot_beeswarm(sg["values"], sg["data"], sg["feature_names"]), width="stretch")
                st.markdown("Lecture : chaque point est un individu ; **à droite**, la variable augmente la probabilité "
                            "de la classe positive ; **couleur rouge** = valeur élevée de la variable.")
                mean_abs = np.abs(np.array(sg["values"], float)).mean(0)
                top_feats = [sg["feature_names"][j] for j in np.argsort(-mean_abs)]
                dep = st.multiselect("Graphiques de dépendance SHAP", sg["feature_names"], default=top_feats[:2],
                                     max_selections=4)
                dcols = st.columns(2) if dep else []
                for i, f in enumerate(dep):
                    dcols[i % 2].plotly_chart(XP.plot_dependence(sg["values"], sg["data"], sg["feature_names"], f),
                                              width="stretch")
                with st.expander("Importance SHAP agrégée par variable d'origine"):
                    st.plotly_chart(XP.plot_importance_by_variable(sg["importance_by_variable"]), width="stretch")
            else:
                st.info("Cliquez sur « Calculer SHAP » pour afficher le summary plot et les dépendances.")

        # ------------------------------- Locale --------------------------------- #
        with xl:
            st.subheader("Expliquer un individu")
            mode = st.radio("Individu à expliquer", ["Individu du jeu de test", "Saisir un individu fictif"],
                            horizontal=True)
            data_in, true_label, go_btn = None, None, False
            if mode == "Individu du jeu de test":
                pos = int(st.number_input("Position dans le jeu de test", 0, tr["n_test"] - 1, 0))
                try:
                    ind = cached(("ind", tid, pos), lambda: xai_sync(
                        tr, "GET", f"/individual/{pos}", lambda s_: EX.get_individual(s_, pos)))
                    data_in, true_label = ind["data"], ind["true_label"]
                    st.dataframe(pd.DataFrame([data_in]), width="stretch", hide_index=True)
                    go_btn = st.button("🔍 Expliquer cet individu", type="primary")
                except Exception as e:
                    show_xai_error(e)
            else:
                data_in, go_btn = render_form(tr["schema"], "xai_form", "🔍 Expliquer cet individu")
                st.caption("Gardez des valeurs cohérentes (par ex. total facturé ≈ ancienneté × facture mensuelle).")

            if go_btn and data_in is not None:
                payload = {"model": mname, "threshold": float(thr), "data": T._py(data_in)}
                try:
                    with st.spinner("Calcul des explications SHAP et LIME…"):
                        res = xai_sync(tr, "POST", "/explain/compare",
                                       lambda s_: EX.compare_local(s_, mname, data_in, thr),
                                       payload=payload, timeout=180)
                    xs["local"] = {"res": res, "true": true_label, "source": tr["source"]}
                except Exception as e:
                    show_xai_error(e)

            loc = xs.get("local")
            if loc:
                res = loc["res"]
                st.divider()
                st.caption(f"Modèle : **{res['model']}** · explications calculées par : **{loc['source']}**")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Probabilité de la classe positive", f"{res['probability']:.1%}")
                r2.metric("Seuil de décision", f"{res['threshold']:.0%}")
                r3.metric("Décision", "Positif" if res["decision"] else "Négatif")
                if loc["true"] is not None:
                    r4.metric("Réalité (jeu de test)", "Positif" if loc["true"] else "Négatif")
                st.info(f"💬 {res['summary']}")

                sh = res["shap"]
                contrib = pd.DataFrame(sh["contributions"])
                if sh["n_other_features"] > 0:
                    contrib = pd.concat([contrib, pd.DataFrame([{
                        "label": f"{sh['n_other_features']} autres variables",
                        "contribution": sh["other_contribution"]}])], ignore_index=True)
                st.plotly_chart(XP.plot_waterfall(contrib, sh["base_value"], sh["units"], res["probability"]),
                                width="stretch")
                if sh["units"] == "log-odds":
                    st.caption("Les contributions sont en log-odds (échelle du modèle) : leur somme avec la valeur de "
                               "base donne le score, converti en probabilité par la fonction logistique.")
                st.plotly_chart(XP.plot_lime(res["lime"]["rules"], "Explication LIME du même individu"),
                                width="stretch")
                st.caption(f"Qualité de l'approximation locale de LIME (R²) : {res['lime']['local_fit_r2']:.2f}")

                cm = res["comparison"]
                st.subheader("SHAP et LIME sont-ils cohérents ?")
                q1, q2 = st.columns(2)
                q1.metric(f"Variables communes dans le top {cm['k']}", f"{cm['overlap']} / {cm['k']}")
                q2.metric("Accord sur le sens de l'effet",
                          "—" if cm["sign_agreement"] is None else f"{cm['sign_agreement']:.0%}")
                ct = pd.DataFrame(cm["table"])[["label", "SHAP", "LIME", "Rang SHAP", "Rang LIME", "Même sens"]]
                show_table(ct.rename(columns={"label": "Variable"}), fmt={"SHAP": "{:+.3f}", "LIME": "{:+.3f}"})
                st.caption("SHAP et LIME sont sur des échelles différentes : on compare les rangs et le sens de "
                           "l'effet, pas les valeurs.")

        # ------------------------------- Équité --------------------------------- #
        with xe:
            st.subheader("Variables sensibles et équité")
            st.caption("Le modèle s'appuie-t-il sur des variables sensibles (sexe, âge, situation familiale…) ? "
                       "Même sans les utiliser directement, d'autres variables peuvent en être des substituts (proxies).")
            default_sens = [c for c in tr["raw_features"] if c in SENSITIVE_HINTS]
            sens = st.multiselect("Variables sensibles à examiner", tr["raw_features"], default=default_sens)
            if sens:
                try:
                    fr_ = cached(("fair", tid, mname, round(thr, 2), tuple(sens)), lambda: xai_sync(
                        tr, "GET", "/fairness", lambda s_: EX.fairness(s_, mname, thr, sens),
                        params={"model": mname, "threshold": thr, "sensitive": ",".join(sens)}))
                    rep = pd.DataFrame(fr_["report"])
                    show_table(rep, fmt={c: "{:.3f}" for c in rep.columns if c not in ("Variable", "Groupe", "Effectif")})
                    st.plotly_chart(XP.plot_fairness(rep), width="stretch")
                    st.markdown("**Écart de parité** : différence maximale du taux de prédictions positives entre groupes.")
                    st.dataframe(pd.DataFrame(fr_["gaps"]).round(3), width="stretch", hide_index=True)
                    sg = xs.get(("shapg", tid, mname))
                    if sg:
                        imp = pd.DataFrame(sg["importance_by_variable"])
                        part = imp[imp["Variable"].isin(sens)]["Part"].sum()
                        st.markdown(f"**Part de l'importance SHAP portée par les variables sensibles : {part:.1%}**")
                    else:
                        st.info("Calculez SHAP (onglet « Explications globales ») pour mesurer le poids de ces variables.")
                except Exception as e:
                    show_xai_error(e)
            else:
                st.info("Sélectionnez au moins une variable sensible.")

# =========================================================================== #
# Prédiction d'un individu (appelle l'API)
# =========================================================================== #
with tab6:
    tr = st.session_state.get("train")
    if not tr:
        st.info("Entraînez d'abord des modèles (onglet 3).")
    else:
        st.subheader("Prédire pour un nouvel individu")
        pm = st.selectbox("Modèle", list(tr["scores"]), key="pred_model")
        data_in, go_pred = render_form(tr["schema"], "pred_form", "🔮 Prédire")
        st.caption("Gardez des valeurs cohérentes (par ex. total facturé ≈ ancienneté × facture mensuelle).")

        if go_pred:
            thr = st.session_state["threshold"]
            try:
                if is_api(tr):
                    out = api_post(f"/sessions/{tr['session_id']}/predict",
                                   {"model": pm, "threshold": float(thr), "data": T._py(data_in)})
                else:
                    out = T.predict_one(st.session_state["local_session"], pm, data_in, thr)
                st.session_state["last_pred"] = out | {"_source": tr["source"]}
            except Exception as e:
                show_xai_error(e)

        out = st.session_state.get("last_pred")
        if out:
            st.divider()
            r1, r2, r3 = st.columns(3)
            r1.metric("Probabilité de la classe positive", f"{out['probability']:.1%}")
            r2.metric("Seuil de décision", f"{out['threshold']:.0%}")
            r3.metric("Décision", "Positif" if out["decision"] else "Négatif")
            st.caption(f"Modèle : {out['model']} · calculé par : {out['_source']}")
            st.progress(min(max(out["probability"], 0.0), 1.0))
