# Prompt 14 — CI CPU précoce et validation d'artifacts

Mettre en place le socle CPU avec prompt 01, puis l'étendre à mesure du code : install,
Ruff, mypy, tests unitaires et intégration CPU, build de l'image API si créée. Cache
dépendances, permissions minimales, aucun téléchargement de poids ni import GPU requis.
Les tests restent propriété de chaque étape ; cette étape consolide la CI.

Valider les artifacts Kaggle joints/importés avec les mêmes schémas/comparateur.
Un contrôle CPU ne constitue pas un benchmark GPU. Kaggle n'est pas un runner GitHub
permanent ; les résultats officiels proviennent de sessions réellement qualifiées.
Un workflow self-hosted GPU et le commentaire PR sont des extensions explicitement
optionnelles, documentées dans OPTIONAL_STRETCH.md ; ne pas les ajouter par défaut.
Conserver result/report/environment et raison structurée des codes 0/1/2.
