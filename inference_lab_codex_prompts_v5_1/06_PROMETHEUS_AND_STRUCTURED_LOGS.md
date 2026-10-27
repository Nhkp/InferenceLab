# Prompt 06 — Observabilité fidèle au streaming et aux sources

Appliquer le contrat. Exposer /metrics sur le gateway : requêtes/succès/erreurs,
durée complète de stream, in-flight et tokens effectivement connus. Distinguer
headers et fin de génération ; décrémenter in-flight à terminaison/annulation.
Durées upstream locales seulement avec frontières explicites ; aucun histogramme
« overhead » produit par soustraction de percentiles ou de durées non alignées.

Labels bornés : model/backend/status_class si utiles. Jamais request_id/run_id libre,
prompt ou hash de prompt en label. Logs JSON corrélés par request_id/run_id, sans contenu.

Prometheus local versionné sous deploy/prometheus, compatible mock sans GPU. Scraper
vLLM seulement sur un hôte accessible et avec noms confirmés pour version qualifiée.
Ne pas supposer un accès réseau du laptop vers Kaggle. Les artifacts restent autonomes.
Séparer métriques client du runner, runtime vLLM, gateway et GPU ; ne pas substituer
un percentile histogramme au percentile client calculé sur les données brutes.
Documenter logs / time series / artifacts et DB d'import. OTel reste inactif.
Tests : durée stream complet, annulation, labels bornés et redaction.
