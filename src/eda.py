"""Exploration (EDA) : tableaux et graphiques Plotly interactifs, utilisés par le notebook et l'onglet « Données »."""
import numpy as np
import pandas as pd
import plotly.express as px

TEMPLATE = "plotly_white"
CLASS_NAMES = {0: "Négatif (0)", 1: "Positif (1)"}


def _cls(y):
    return pd.Series(y).map(CLASS_NAMES)


def missing_values_table(df_raw):
    """NaN classiques + valeurs vides cachées (chaînes vides ou espaces) par colonne."""
    nan = df_raw.isna().sum()
    obj = df_raw.select_dtypes(include=["object", "string"])
    blank = obj.apply(lambda s: s.fillna("x").astype(str).str.strip().eq("").sum()).reindex(df_raw.columns).fillna(0).astype(int)
    out = pd.DataFrame({"NaN": nan, "Vides cachés": blank})
    out["Total manquants"] = out["NaN"] + out["Vides cachés"]
    out["% manquants"] = (100 * out["Total manquants"] / len(df_raw)).round(2)
    return out.sort_values("Total manquants", ascending=False)


def class_balance(df, target):
    vc = df[target].value_counts().sort_index()
    return pd.DataFrame({"Classe": [CLASS_NAMES.get(i, str(i)) for i in vc.index],
                         "Effectif": vc.values, "Proportion": (vc.values / vc.sum()).round(4)})


def plot_class_balance(df, target):
    t = class_balance(df, target)
    fig = px.bar(t, x="Classe", y="Effectif", color="Classe", template=TEMPLATE,
                 text=[f"{e} ({p:.1%})" for e, p in zip(t["Effectif"], t["Proportion"])],
                 title="Répartition de la cible")
    fig.update_layout(showlegend=False)
    return fig


def plot_numeric_distribution(df, col, target):
    fig = px.histogram(df, x=col, color=_cls(df[target]), nbins=40, barmode="overlay", opacity=0.65,
                       marginal="box", template=TEMPLATE, title=f"Distribution de {col} selon la cible",
                       labels={"color": "Classe"})
    return fig


def plot_box_by_target(df, col, target):
    return px.box(df, x=_cls(df[target]), y=col, color=_cls(df[target]), template=TEMPLATE,
                  title=f"{col} selon la cible", labels={"x": "Classe", "color": "Classe"})


def plot_correlation(df, target, num_cols=None):
    if num_cols is None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cols = list(dict.fromkeys(list(num_cols) + [target]))
    corr = df[cols].corr()
    return px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     template=TEMPLATE, title="Matrice de corrélation (variables numériques et cible)")


def high_correlations(df, num_cols, threshold=0.7):
    """Paires de variables numériques fortement corrélées (redondance)."""
    corr = df[list(num_cols)].corr().abs()
    pairs = [(a, b, round(corr.loc[a, b], 3)) for i, a in enumerate(corr.columns)
             for b in corr.columns[i + 1:] if corr.loc[a, b] >= threshold]
    return pd.DataFrame(pairs, columns=["Variable A", "Variable B", "|corrélation|"])


def rate_by_category(df, col, target, bins=8):
    """Taux de la classe positive par modalité (les variables numériques sont découpées en quantiles)."""
    s = df[col]
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
        s = pd.qcut(s, q=min(bins, s.nunique()), duplicates="drop").astype(str)
    g = df.groupby(s.astype(str), observed=True)[target].agg(["mean", "count"]).reset_index()
    g.columns = ["Modalité", "Taux positif", "Effectif"]
    return g


def plot_rate_by_category(df, col, target, bins=8):
    g = rate_by_category(df, col, target, bins)
    overall = df[target].mean()
    fig = px.bar(g, x="Modalité", y="Taux positif", hover_data=["Effectif"], template=TEMPLATE,
                 text=g["Taux positif"].map("{:.1%}".format), title=f"Taux de la classe positive par {col}")
    fig.add_hline(y=overall, line_dash="dash", line_color="red", annotation_text=f"Moyenne {overall:.1%}")
    fig.update_yaxes(tickformat=".0%")
    return fig


def rate_spread(df, target, cols):
    """Classement des variables par écart de taux positif entre modalités (pouvoir discriminant)."""
    rows = []
    for c in cols:
        g = rate_by_category(df, c, target)
        g = g[g["Effectif"] >= 30]
        if len(g) > 1:
            rows.append((c, g["Taux positif"].max() - g["Taux positif"].min()))
    return pd.DataFrame(rows, columns=["Variable", "Écart max"]).sort_values("Écart max", ascending=False)


def plot_rate_spread(df, target, cols):
    t = rate_spread(df, target, cols)
    fig = px.bar(t, x="Écart max", y="Variable", orientation="h", template=TEMPLATE,
                 title="Variables classées par écart de taux positif entre modalités")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def plot_crosstab_rate(df, row, col, target, bins=5):
    def disc(s):
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
            return pd.qcut(s, q=bins, duplicates="drop").astype(str)
        return s.astype(str)
    piv = df.assign(_r=disc(df[row]), _c=disc(df[col])).pivot_table(index="_r", columns="_c", values=target,
                                                                     aggfunc="mean", observed=True)
    return px.imshow(piv, text_auto=".1%", color_continuous_scale="Reds", aspect="auto", template=TEMPLATE,
                     labels={"x": col, "y": row, "color": "Taux positif"},
                     title=f"Taux de la classe positive : {row} × {col}")
