# One-pager : Prédire le départ des clients (Telco)
*(1 page PDF : phrases courtes, 1 ou 2 graphiques)*

## 1. Problème métier
Quels clients risquent de résilier ? Coût d'un départ vs coût d'une action de rétention.

## 2. Données
7 043 clients, 20 variables, 26,5 % de départs. Problèmes traités : `TotalCharges` (11 vides, clients à ancienneté 0),
classes déséquilibrées (73/27), redondance tenure / TotalCharges.

## 3. Méthode
Split stratifié, imputation, OneHot, normalisation (ajustés sur le train), 5 modèles, RandomizedSearchCV (AUC),
pondération des classes, seuil de décision réglable.

## 4. Meilleur modèle et résultats
| Modèle | AUC | PR-AUC | Rappel | Précision | F1 |
|---|---|---|---|---|---|
| … | … | … | … | … | … |
Seuil retenu : … (justification : rappel vs précision, coût des erreurs). Matrice de confusion : …

## 5. Compromis performance / simplicité
Écart d'AUC entre le meilleur modèle et la régression logistique : … → recommandation au décideur.

## 6. Recommandations et limites
Cibler les clients à haut risque (contrat mensuel, faible ancienneté…) ; limites : un seul jeu de données,
un seul découpage, association ≠ causalité, seuil dépendant des coûts réels.
