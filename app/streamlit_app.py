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
from src import plots
from src import preprocess as P
from src import trainer as T
from src.api_client import api_get, api_health, api_post, get_api_url

st.set_page_config(page_title="Classification interactive", page_icon="📊", layout="wide")
st.session_state.setdefault("threshold", 0.5)

IMPUTERS = {"Médiane": "median", "Moyenne": "mean", "Zéro (0)": "zero"}


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
    for k in ["prep", "train", "local_session", "last_pred"]:
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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["1️⃣ Données", "2️⃣ Préparation", "3️⃣ Modélisation", "4️⃣ Évaluation",
                                        "5️⃣ Prédiction"])

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
                for k in ["train", "local_session", "last_pred"]:
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
                    "scores": {m: np.array(v) for m, v in result["scores"].items()},
                    "y_test": np.array(result["y_test"]), "best_params": result["best_params"],
                    "cv_scores": result["cv_scores"], "n_train": result["n_train"], "n_test": result["n_test"],
                    "schema": result["schema"], "raw_features": result["raw_features"]}
                st.session_state.pop("last_pred", None)
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
# Prédiction d'un individu (appelle l'API)
# =========================================================================== #
with tab5:
    tr = st.session_state.get("train")
    if not tr:
        st.info("Entraînez d'abord des modèles (onglet 3).")
    else:
        st.subheader("Prédire pour un nouvel individu")
        pm = st.selectbox("Modèle", list(tr["scores"]), key="pred_model")
        schema = tr["schema"]
        with st.form("pred_form"):
            fcols = st.columns(3)
            data_in = {}
            for i, (c, sc) in enumerate(schema.items()):
                with fcols[i % 3]:
                    if sc["type"] == "cat":
                        ch = sc["choices"]
                        data_in[c] = st.selectbox(c, ch, index=ch.index(sc["mode"]))
                    elif float(sc["min"]).is_integer() and float(sc["max"]).is_integer():
                        data_in[c] = int(st.number_input(c, int(sc["min"]), int(sc["max"]), int(sc["median"])))
                    else:
                        data_in[c] = float(st.number_input(c, float(sc["min"]), float(sc["max"]), float(sc["median"])))
            go_pred = st.form_submit_button("🔮 Prédire", type="primary")
        st.caption("Gardez des valeurs cohérentes (par ex. total facturé ≈ ancienneté × facture mensuelle).")

        if go_pred:
            thr = st.session_state["threshold"]
            try:
                if tr["source"] == "API FastAPI":
                    out = api_post(f"/sessions/{tr['session_id']}/predict",
                                   {"model": pm, "threshold": float(thr), "data": T._py(data_in)})
                else:
                    out = T.predict_one(st.session_state["local_session"], pm, data_in, thr)
                st.session_state["last_pred"] = out | {"_source": tr["source"]}
            except requests.RequestException as e:
                st.error(f"API injoignable ({type(e).__name__}). Relancez l'API ou ré-entraînez en local.")
            except RuntimeError as e:
                st.error(f"{e}")

        out = st.session_state.get("last_pred")
        if out:
            st.divider()
            r1, r2, r3 = st.columns(3)
            r1.metric("Probabilité de la classe positive", f"{out['probability']:.1%}")
            r2.metric("Seuil de décision", f"{out['threshold']:.0%}")
            r3.metric("Décision", "Positif" if out["decision"] else "Négatif")
            st.caption(f"Modèle : {out['model']} · calculé par : {out['_source']}")
            st.progress(min(max(out["probability"], 0.0), 1.0))
