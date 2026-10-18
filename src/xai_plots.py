"""Graphiques du module 5 (Plotly, interactifs) : construits à partir des résultats de src/explain_service.py."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TEMPLATE = "plotly_white"
RED, BLUE = "#d62728", "#1f77b4"
SCALE = [[0, "#1f77b4"], [0.5, "#9467bd"], [1, "#d62728"]]


def plot_native_importance(df, top=15):
    df = pd.DataFrame(df)
    d = df.head(top).iloc[::-1]
    kind = df["Type"].iloc[0]
    signed = kind.startswith("Coefficient")
    colors = [RED if v > 0 else BLUE for v in d["Importance"]] if signed else BLUE
    fig = go.Figure(go.Bar(x=d["Importance"], y=d["Variable"], orientation="h", marker_color=colors))
    fig.update_layout(template=TEMPLATE, title=f"Importance propre au modèle : {kind}",
                      xaxis_title="Rouge = augmente le risque, bleu = le diminue" if signed else "Importance",
                      height=max(350, 24 * len(d) + 120))
    return fig


def plot_permutation_comparison(perm, top=10):
    perm = pd.DataFrame(perm)
    order = perm.groupby("Variable")["Importance"].mean().sort_values(ascending=False).head(top).index
    d = perm[perm["Variable"].isin(order)]
    fig = px.bar(d, x="Importance", y="Variable", color="Modèle", barmode="group", orientation="h",
                 template=TEMPLATE, title="Importance par permutation (baisse d'AUC), comparée entre modèles")
    fig.update_layout(yaxis={"categoryorder": "array", "categoryarray": list(order)[::-1]},
                      height=max(400, 55 * len(order) + 150))
    return fig


def plot_beeswarm(values, data, feature_names, max_display=15, seed=0):
    """Équivalent interactif du « SHAP summary plot » : un point par individu et par variable.
    Position horizontale = effet sur la prédiction ; couleur = valeur de la variable (rouge = élevée)."""
    values, data = np.asarray(values, float), np.asarray(data, float)
    order = list(np.argsort(-np.abs(values).mean(0))[:max_display])
    rng = np.random.default_rng(seed)
    xs, ys, cs, cd = [], [], [], []
    for rank, j in enumerate(order):
        v = data[:, j]
        lo, hi = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
        if hi - lo < 1e-12:
            lo, hi = np.nanmin(v), np.nanmax(v)
        norm = np.clip((v - lo) / (hi - lo), 0, 1) if hi > lo else np.full(len(v), 0.5)
        xs.extend(values[:, j]); cs.extend(norm)
        ys.extend((len(order) - 1 - rank) + rng.uniform(-0.3, 0.3, len(v)))
        cd.extend([[feature_names[j], float(x)] for x in v])
    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="markers", customdata=cd,
        marker=dict(size=5, color=cs, colorscale=SCALE, cmin=0, cmax=1, opacity=0.8,
                    colorbar=dict(title="Valeur", tickvals=[0, 1], ticktext=["Faible", "Élevée"],
                                  thickness=12, len=0.5)),
        hovertemplate="%{customdata[0]} = %{customdata[1]:.4g}<br>Effet SHAP : %{x:+.3f}<extra></extra>"))
    fig.add_vline(x=0, line_color="#999", line_width=1)
    fig.update_yaxes(tickvals=list(range(len(order))), ticktext=[feature_names[j] for j in order][::-1])
    fig.update_layout(template=TEMPLATE, title="SHAP summary plot : quelles variables pèsent, et dans quel sens",
                      xaxis_title="Effet sur la prédiction (à droite : augmente le risque)",
                      height=max(420, 30 * len(order) + 140), showlegend=False)
    return fig


def auto_color_feature(values, data, j):
    """Variable dont les valeurs expliquent le mieux ce que la variable j ne fait pas seule (interaction)."""
    x, s = data[:, j], values[:, j]
    if np.nanstd(x) < 1e-12:
        return None
    resid = s - np.polyval(np.polyfit(x, s, 1), x)
    best, best_c = None, 0.0
    for k in range(data.shape[1]):
        if k == j or np.nanstd(data[:, k]) < 1e-12:
            continue
        c = abs(np.corrcoef(data[:, k], resid)[0, 1])
        if np.isfinite(c) and c > best_c:
            best, best_c = k, c
    return best


def plot_dependence(values, data, feature_names, feature):
    """Effet SHAP d'une variable selon sa valeur, coloré par la variable qui interagit le plus."""
    values, data = np.asarray(values, float), np.asarray(data, float)
    j = feature_names.index(feature)
    k = auto_color_feature(values, data, j)
    marker = dict(size=6, opacity=0.8)
    title = f"Dépendance SHAP : {feature}"
    if k is not None:
        marker.update(color=data[:, k], colorscale=SCALE,
                      colorbar=dict(title=feature_names[k], thickness=12, len=0.6))
        title += f" (couleur : {feature_names[k]})"
    fig = go.Figure(go.Scatter(x=data[:, j], y=values[:, j], mode="markers", marker=marker,
                               hovertemplate=f"{feature} = %{{x:.4g}}<br>Effet SHAP : %{{y:+.3f}}<extra></extra>"))
    fig.add_hline(y=0, line_color="#999", line_width=1)
    fig.update_layout(template=TEMPLATE, title=title, xaxis_title=feature, yaxis_title="Effet SHAP",
                      height=380)
    return fig


def plot_importance_by_variable(imp, top=15):
    d = pd.DataFrame(imp).head(top).iloc[::-1]
    fig = px.bar(d, x="Importance SHAP moyenne", y="Variable", orientation="h", template=TEMPLATE,
                 title="Importance SHAP moyenne par variable d'origine")
    fig.update_layout(height=max(350, 24 * len(d) + 120))
    return fig


def plot_waterfall(contrib, base_value, units, proba, title=None):
    """Waterfall : on part de la valeur moyenne, chaque variable ajoute (rouge) ou retire (bleu)."""
    c = pd.DataFrame(contrib)
    labels = ["Valeur moyenne (base)"] + list(c["label"])
    values = [base_value] + [float(v) for v in c["contribution"]]
    measure = ["absolute"] + ["relative"] * len(c) + ["total"]
    labels.append(f"Prédiction (proba {proba:.0%})")
    values.append(0)
    texts = [f"{base_value:+.2f}"] + [f"{v:+.2f}" for v in values[1:-1]] + [""]
    fig = go.Figure(go.Waterfall(
        orientation="h", y=labels, x=values, measure=measure, text=texts, textposition="outside",
        increasing=dict(marker=dict(color=RED)), decreasing=dict(marker=dict(color=BLUE)),
        totals=dict(marker=dict(color="#555555")), connector=dict(line=dict(color="#aaaaaa"))))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(template=TEMPLATE, title=title or "SHAP waterfall : contribution de chaque variable",
                      xaxis_title=f"Contribution ({units}) : rouge = augmente le risque, bleu = le diminue",
                      height=max(380, 36 * len(labels) + 130), margin=dict(l=10, r=80))
    return fig


def plot_lime(table, title="Explication LIME"):
    d = pd.DataFrame(table)
    d = d.reindex(d["weight"].abs().sort_values().index)
    fig = go.Figure(go.Bar(x=d["weight"], y=d["rule"], orientation="h",
                           marker_color=[RED if w > 0 else BLUE for w in d["weight"]]))
    fig.update_layout(template=TEMPLATE, title=title,
                      xaxis_title="Poids LIME : rouge = augmente le risque, bleu = le diminue",
                      height=max(350, 34 * len(d) + 120), margin=dict(l=10))
    return fig


def plot_fairness(report):
    d = pd.DataFrame(report)
    return px.bar(d, x="Variable", y="Taux prédit positif", color="Groupe", barmode="group", template=TEMPLATE,
                  title="Taux de prédictions positives par groupe", hover_data=["Effectif"])
