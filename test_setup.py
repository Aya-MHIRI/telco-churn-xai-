"""Vérifie que l'environnement et le chargement des données fonctionnent."""
from src.data import load_data, clean

df = clean(load_data())
print(df.shape)                          # attendu : (7043, 20)
print(round(df["Churn"].mean(), 3))      # attendu : ~0.265
print(df["TotalCharges"].isna().sum())   # attendu : 11
