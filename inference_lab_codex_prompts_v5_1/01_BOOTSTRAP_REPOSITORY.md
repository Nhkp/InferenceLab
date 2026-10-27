# Prompt 01 — Bootstrap minimal, après qualification initiale

Lire les règles 00, le PDD et le contrat. Inspecter le résultat de 00B ; une qualification
GPU non exécutée reste ouverte, mais ne bloque pas les fondations CPU indépendantes.

Créer `src/inference_lab/{gateway,benchmark,registry,telemetry,storage}`, config,
`kernel_lab/rmsnorm`, experiments/datasets, dashboards, deploy/{docker,prometheus,grafana},
kaggle, docs, tests/{unit,integration}, scripts, results et .github/workflows.
Paquet installable Python 3.12 avec typage, Pydantic v2/pydantic-settings, FastAPI,
httpx, SQLAlchemy 2/Alembic/psycopg, prometheus-client, structlog ; outils pytest,
pytest-asyncio, Ruff et mypy. Dépendances/extras GPU distincts, versions issues de 00B.
L'installation CPU ne doit importer vLLM/Triton ni télécharger de poids.

Créer uniquement `GET /health` avec test et un faux serveur HTTP streaming testable
sans modèle pour les étapes suivantes. Ne pas construire de métier supplémentaire.
Makefile : install, lint, format, typecheck, test, run, up, down. Compose CPU/mock.
README basé sur le template, état réel/roadmap distingués. Ne pas annoncer le GPU validé.
Mettre en place maintenant les contrôles CPU du prompt 14 ; ne pas attendre la fin.

Acceptation : installation CPU propre, health, lint, mypy, tests sans CUDA/poids.
