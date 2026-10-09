"""Graphiques d'évaluation (Plotly), construits à partir des scores des modèles."""
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.models import confusion_counts, threshold_table

TEMPLATE = "plotly_white"


def plot_roc(scores, y_true):
    fig = go.Figure()
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={roc_auc_score(y_true, s):.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Hasard", line=dict(dash="dash", color="grey")))
    fig.update_layout(template=TEMPLATE, title="Courbes ROC superposées", xaxis_title="Taux de faux positifs",
                      yaxis_title="Taux de vrais positifs (rappel)", legend=dict(x=0.4, y=0.05))
    return fig


def plot_pr(scores, y_true):
    fig = go.Figure()
    for name, s in scores.items():
        prec, rec, _ = precision_recall_curve(y_true, s)
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                 name=f"{name} (PR-AUC={average_precision_score(y_true, s):.3f})"))
    base = float(np.mean(y_true))
    fig.add_hline(y=base, line_dash="dash", line_color="grey", annotation_text=f"Hasard ({base:.2f})")
    fig.update_layout(template=TEMPLATE, title="Courbes précision-rappel", xaxis_title="Rappel",
                      yaxis_title="Précision")
    return fig


def plot_confusion(y_true, proba, threshold=0.5, title=None):
    c = confusion_counts(y_true, proba, threshold)
    z = [[c["VN"], c["FP"]], [c["FN"], c["VP"]]]
    txt = [[f"VN<br>{c['VN']}", f"FP (fausse alerte)<br>{c['FP']}"],
           [f"FN (raté)<br>{c['FN']}", f"VP<br>{c['VP']}"]]
    fig = go.Figure(go.Heatmap(z=z, x=["Prédit : négatif", "Prédit : positif"], y=["Réel : négatif", "Réel : positif"],
                               text=txt, texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(template=TEMPLATE, title=title or f"Matrice de confusion (seuil {threshold:.2f})")
    return fig


def plot_threshold_curve(y_true, proba, current=None):
    t = threshold_table(y_true, proba)
    fig = go.Figure()
    for col in ["Précision", "Rappel", "F1"]:
        fig.add_trace(go.Scatter(x=t["Seuil"], y=t[col], mode="lines", name=col))
    if current is not None:
        fig.add_vline(x=current, line_dash="dash", line_color="red", annotation_text=f"Seuil {current:.2f}")
    fig.update_layout(template=TEMPLATE, title="Effet du seuil de décision", xaxis_title="Seuil", yaxis_title="Valeur")
    return fig
