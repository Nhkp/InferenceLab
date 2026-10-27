# Prompt 13 — Audit des tests, construits dès chaque étape

Ne pas attendre ce prompt pour écrire les tests nécessaires. Auditer et compléter :

- unit : schémas, defaults, plans/hashes, stats/percentiles, policies/quatre verdicts ;
- streaming : fragments, rôle seul, usage, multi-token, single chunk, fin abrupte,
  erreur après HTTP 200, timeouts, progression sans buffering et annulation upstream ;
- intégration CPU : FastAPI + SSE fake, campagne sans DB, recalcul raw -> summary,
  import PostgreSQL idempotent/conflit/rollback, migrations et métriques stream complet ;
- comparabilité : changement autorisé, différences cachées, GPU utilisés, volume,
  source de métrique et versions incompatibles ;
- GPU : génération réelle, configuration/backend effectifs, annulation, qualification
  AWQ/TP/kernel quand disponible ; aucune validation GPU obtenue par simple import.

Marks : unit, integration, gpu, slow. CI CPU : `pytest -m "not gpu and not slow"`.
GPU : `pytest -m gpu`, exécution explicite ; rapporter séparément passes/skips/échecs.
Pas de poids téléchargés dans les tests CPU. Fixtures identifiées synthétiques.
Documenter limites de couverture et commandes dans docs/testing.md.
