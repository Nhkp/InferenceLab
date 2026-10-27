# Prompt 12 — Analyse Grafana après MVP système

Provisionner Grafana local sans dépendance des campagnes Kaggle à une connexion
distante. Mock/hôte accessible pour dashboards live ; artifacts Kaggle pour rapport
offline. Ne pas faire croire à des métriques live inexistantes.
Panels API : requêtes, erreurs terminales, latence stream complet, in-flight et tokens.
Panels vLLM : noms vérifiés sur version qualifiée, TTFT/TPOT/ITL runtime distincts,
running/waiting, KV cache et prefill/decode disponibles.
Panels GPU : métriques réellement collectées, absence/limites signalées ; DCGM facultatif.
Les percentiles Prometheus ne remplacent pas les agrégats client du runner/gates.
Pas de panel gateway overhead par soustraction de p95 ; nommer écarts de chemins et
durées locales selon leurs frontières. Conserver labels bornés et unités explicites.
Chaque panel répond à une question système ; pas de nouveau frontend ni design excessif.
