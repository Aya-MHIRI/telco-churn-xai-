# Module 5 : explicabilité (XAI)

Objectif : comprendre **pourquoi** un modèle prédit ce qu'il prédit, globalement (le modèle entier) et localement (un client).

## Lire les graphiques
| Graphique | Question | Lecture |
|---|---|---|
| Importance propre au modèle | Quelles variables le modèle utilise-t-il ? | Régression logistique : coefficient > 0 (rouge) = augmente le risque. Arbres : importance par impureté (sans signe). |
| Importance par permutation | Que se passe-t-il si on casse une variable ? | Baisse d'AUC quand on mélange la variable : plus elle est grande, plus le modèle en dépend. Comparable entre modèles. |
| SHAP summary (beeswarm) | Quelles variables pèsent, et dans quel sens ? | Un point = un client. À droite : augmente le risque. Rouge = valeur élevée de la variable. |
| SHAP dépendance | Comment l'effet change-t-il avec la valeur ? | Ex. : l'effet de `tenure` diminue quand l'ancienneté augmente. |
| SHAP waterfall | Pourquoi ce client ? | On part de la valeur moyenne ; chaque variable ajoute (rouge) ou retire (bleu). |
| LIME | Même question, autre méthode | Petit modèle simple construit autour du client ; poids de chaque règle. |
| Cohérence SHAP / LIME | Les deux méthodes disent-elles la même chose ? | Variables communes dans le top 5 et accord sur le sens de l'effet. |
| Équité | Le modèle s'appuie-t-il sur des variables sensibles ? | Taux de prédictions positives par groupe, écart de parité, part de l'importance SHAP des variables sensibles. |

SHAP et LIME décrivent le **modèle**, pas la réalité : une influence est une association, pas une cause.

## Où est le code
- `src/xai.py` : calculs (permutation, SHAP, LIME, comparaison, phrase de synthèse, équité).
- `src/explain_service.py` : fonctions appelées par l'API (`/sessions/{id}/explain/*`, `/fairness`) et par l'appli en mode local.
- `src/xai_plots.py` : graphiques Plotly interactifs.
- `notebooks/03_explicabilite.ipynb` : le même parcours en notebook.

## Texte à ajouter au README (si votre README a déjà vos liens)
Dans « Fonctionnalités » : `5. Explicabilité (XAI) : importances (native, permutation), SHAP (beeswarm, dépendance, waterfall), LIME, comparaison SHAP/LIME, phrase de synthèse, équité`.
Dans « API » : routes `/sessions/{id}/explain/native`, `/explain/permutation`, `/explain/shap-global`, `/explain/shap`, `/explain/lime`, `/explain/compare`, `/fairness`, `/individual/{i}`.
Dans « Structure » : `src/xai.py, explain_service.py, xai_plots.py` et `notebooks/03_explicabilite.ipynb`.
