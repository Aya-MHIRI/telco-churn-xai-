"""
Prétraitement : fonctions paramétrables, réutilisées par le notebook, l'API et l'application.

Ordre (pour éviter toute fuite de données) :
  1. sélection des variables + nettoyage SANS apprentissage (regroupement de modalités, features simples)
  2. split train/test stratifié
  3. ColumnTransformer (imputation + encodage + normalisation) ajusté sur le TRAIN uniquement
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"]


# --------------------------------------------------------------------------- #
# 1. Nettoyage / features (sans apprentissage)
# --------------------------------------------------------------------------- #
def simplify_categories(df):
    """Regroupe 'No internet service' et 'No phone service' en 'No' (modalités redondantes)."""
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category", "string"]).columns:
        df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})
    return df


def add_features(df):
    """Variables simples et explicables (activées seulement si les colonnes existent) :
    - nb_services    : nombre de services optionnels souscrits (mesure l'attachement)
    - has_protection : a OnlineSecurity ou TechSupport
    """
    df = df.copy()
    present = [c for c in SERVICES if c in df.columns]
    if present:
        df["nb_services"] = (df[present] == "Yes").sum(axis=1)
    if {"OnlineSecurity", "TechSupport"}.issubset(df.columns):
        df["has_protection"] = ((df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes")).astype(int)
    return df


def engineer_input(df, simplify=True, engineer=True):
    """Applique le nettoyage sans apprentissage. Même fonction à l'entraînement et à l'inférence (API)."""
    if simplify:
        df = simplify_categories(df)
    if engineer:
        df = add_features(df)
    return df


def encode_target(y, positive_label=None):
    """Transforme la cible en 0/1. Déjà 0/1 : inchangée. Sinon, positive_label = classe d'intérêt."""
    y = pd.Series(y)
    if set(y.dropna().unique()) <= {0, 1}:
        return y
    if positive_label is None:
        raise ValueError(f"Cible non binaire : précisez la classe positive parmi {list(y.dropna().unique())[:10]}.")
    return (y == positive_label).astype(int).where(y.notna())


# --------------------------------------------------------------------------- #
# 2. Types de variables et split
# --------------------------------------------------------------------------- #
def get_feature_types(df, features):
    """Sépare variables numériques et catégorielles (selon le dtype)."""
    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in features if c not in num_cols]
    return num_cols, cat_cols


def split_data(df, target, features=None, test_size=0.2, seed=42):
    """Split train/test STRATIFIÉ (garde la proportion de la classe positive)."""
    if features is None:
        features = [c for c in df.columns if c != target]
    return train_test_split(df[features], df[target], test_size=test_size, stratify=df[target], random_state=seed)


# --------------------------------------------------------------------------- #
# 3. Transformateur
# --------------------------------------------------------------------------- #
def build_preprocessor(num_cols, cat_cols, imputer="median", scale=True):
    """
    imputer : 'median' | 'mean' | 'zero'  (valeurs manquantes numériques)
    scale   : StandardScaler (indispensable pour régression logistique et KNN)
    Catégorielles : modalité la plus fréquente + OneHot (drop='if_binary' : Yes/No -> 1 colonne).
    """
    if imputer == "zero":
        num_imp = SimpleImputer(strategy="constant", fill_value=0)
    elif imputer in ("median", "mean"):
        num_imp = SimpleImputer(strategy=imputer)
    else:
        raise ValueError("imputer doit être 'median', 'mean' ou 'zero'")

    num_steps = [("imputer", num_imp)]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)),
    ]
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline(num_steps), list(num_cols)))
    if cat_cols:
        transformers.append(("cat", Pipeline(cat_steps), list(cat_cols)))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def get_feature_names(preprocessor):
    """Noms des colonnes après transformation (pour SHAP / LIME / graphiques)."""
    return list(preprocessor.get_feature_names_out())


def transform_to_df(preprocessor, X):
    """Applique un préprocesseur déjà ajusté et renvoie un DataFrame aux colonnes nommées."""
    arr = preprocessor.transform(X)
    return pd.DataFrame(arr, columns=get_feature_names(preprocessor), index=X.index)


# --------------------------------------------------------------------------- #
# Raccourci : tout en un appel
# --------------------------------------------------------------------------- #
def prepare(df, target="Churn", features=None, test_size=0.2, imputer="median",
            scale=True, simplify=True, engineer=True, seed=42):
    """
    df : données nettoyées, cible en 0/1.
    Renvoie un dict : X_train, X_test, y_train, y_test, preprocessor (ajusté sur le train), colonnes...
    """
    if features is None:
        features = [c for c in df.columns if c != target]
    raw_features = list(features)

    d = df[raw_features + [target]].dropna(subset=[target]).copy()
    d = engineer_input(d, simplify, engineer)          # engineering APRÈS sélection : pas de variable écartée
    feats = [c for c in d.columns if c != target]

    X_train, X_test, y_train, y_test = split_data(d, target, feats, test_size, seed)
    num_cols, cat_cols = get_feature_types(d, feats)
    pre = build_preprocessor(num_cols, cat_cols, imputer=imputer, scale=scale)
    pre.fit(X_train)                                     # ajusté sur le TRAIN uniquement

    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train.astype(int), "y_test": y_test.astype(int),
        "preprocessor": pre, "num_cols": num_cols, "cat_cols": cat_cols,
        "features": feats, "raw_features": raw_features, "target": target,
        "feature_names": get_feature_names(pre),
        "settings": {"test_size": test_size, "imputer": imputer, "scale": scale,
                     "simplify": simplify, "engineer": engineer, "seed": seed},
    }
