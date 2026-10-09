"""Chargement et nettoyage des données."""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
LOCAL_PATH = os.path.join(ROOT, "data", CSV_NAME)
KAGGLE_DATASET = "blastchar/telco-customer-churn"


def load_data():
    """Télécharge automatiquement le jeu Telco (kagglehub). Plan B : fichier local dans data/."""
    try:
        import kagglehub

        path = kagglehub.dataset_download(KAGGLE_DATASET)
        return pd.read_csv(os.path.join(path, CSV_NAME))
    except Exception as e:  # réseau, authentification, paquet absent...
        print(f"Téléchargement automatique impossible ({type(e).__name__}) -> fichier local.")
        if not os.path.exists(LOCAL_PATH):
            raise FileNotFoundError(
                f"Le téléchargement a échoué et {LOCAL_PATH} est absent. "
                "Placez le CSV Telco dans le dossier data/."
            ) from e
        return pd.read_csv(LOCAL_PATH)


def clean(df):
    """Nettoyage spécifique Telco (utilisé par le notebook et test_setup.py)."""
    df = df.copy()
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def generic_clean(df, numeric_threshold=0.9):
    """Nettoyage générique, valable pour n'importe quel CSV :
    - supprime les espaces autour des textes, transforme les chaînes vides en NaN
    - convertit en numérique les colonnes texte qui sont en réalité des nombres (ex. TotalCharges)
    - supprime les colonnes identifiant (texte, valeurs toutes uniques)
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].astype("object").where(df[col].notna(), None)
        s = s.map(lambda v: v.strip() if isinstance(v, str) else v)
        s = s.replace("", np.nan)
        num = pd.to_numeric(s, errors="coerce")
        non_null = s.notna().sum()
        if non_null > 0 and num.notna().sum() >= numeric_threshold * non_null:
            df[col] = num
        else:
            df[col] = s
    id_cols = [
        c for c in df.select_dtypes(include=["object", "string"]).columns
        if len(df) > 50 and df[c].nunique(dropna=True) == len(df)
    ]
    return df.drop(columns=id_cols)
