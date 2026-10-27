# InferenceLab — Pack Codex v5, Kaggle/T4

## Démarrage recommandé — v5.1

Commence par `START_HERE.md`. La première étape réelle est `00B_T4_COMPATIBILITY_QUALIFICATION.md`, avant le bootstrap complet.


**PDD v1.2 — 17 octobre 2026 — révision pré-implémentation.**
Ce pack corrige les dix Must fix de la revue. Il contient des spécifications et prompts,
pas une implémentation ni une preuve de compatibilité GPU.

## Documents à lire

1. [PROJECT_DESIGN_DOCUMENT.md](PROJECT_DESIGN_DOCUMENT.md) : scope et décisions v1.2.
2. [BENCHMARK_CONTRACT.md](BENCHMARK_CONTRACT.md) : métriques, YAML, artifacts et verdicts normatifs.
3. [00_GLOBAL_RULES.md](00_GLOBAL_RULES.md) : cadre Codex permanent.
4. [SHOULD_IMPROVE.md](SHOULD_IMPROVE.md) : six recommandations non bloquantes, à sélectionner.
5. [OPTIONAL_STRETCH.md](OPTIONAL_STRETCH.md) : extensions existantes conditionnelles.
6. [CHANGELOG_V5.md](CHANGELOG_V5.md) : traçabilité M1–M10 et validations du pack.

PDD/contrat priment sur tout exemple ou ancien prompt. Ne pas implémenter les should
et optional par simple lecture de leurs documents.

## Décisions structurantes

- Runner HTTP externe, vLLM séparé et gateway SSE mince ; trafic loopback sur Kaggle.
- T4/FP16/TP1 en référence ; qualification de versions/backend avant claims GPU.
- Un modèle principal ~3B, AWQ à ressources identiques, TP même checkpoint/dtype.
- Charge fermée à concurrence constante, sorties contrôlées, caches/config effectifs fixés.
- Artifacts recalculables autonomes ; PostgreSQL importe l'historique localement.
- Gates avec PASS/FAIL/INCONCLUSIVE/INVALID ; pas de chiffres inventés ni p99 certifié par smoke.
- Kernel Lab indépendant requis pour portfolio complet ; intégration runtime optionnelle.

## Ordre d'exécution et dépendances

Les numéros historiques sont conservés pour retrouver les prompts ; suivre cet ordre,
et non un tri alphabétique. Une étape = scope limité, validation puis commit des seuls
fichiers pertinents. Les tests et la documentation accompagnent chaque étape.

| Ordre | Prompt | Condition / objectif |
| --- | --- | --- |
| 1 | [00](00_GLOBAL_RULES.md) puis [00B](00B_T4_COMPATIBILITY_QUALIFICATION.md) | Qualification initiale ; sans GPU, état ouvert explicite |
| 2 | [01](01_BOOTSTRAP_REPOSITORY.md) et socle [14](14_GITHUB_ACTIONS_CI.md) | Bootstrap/CI CPU |
| 3 | [02](02_ARCHITECTURE_AND_CONFIG.md) | Contrats avant stockage et métriques |
| 4 | [03](03_VLLM_SERVING.md) | Serving direct et faux SSE testable |
| 5 | [08](08_LOAD_GENERATOR.md) | Mesure directe et données brutes |
| 6 | [09](09_BENCHMARK_ORCHESTRATOR.md) | Campagnes par point/répétition |
| 7 | [10](10_PERFORMANCE_REGRESSION_GATES.md) | Comparaisons sur fixtures puis mesures qualifiées |
| 8 | [04](04_API_GATEWAY_AND_REQUEST_LOGGING.md) | Chemin gateway streaming |
| 9 | [07](07_GPU_MONITORING.md), [07B](07B_KAGGLE_GPU_ENVIRONMENT.md) | Collecte/runbook Kaggle finalisés |
| 10 | [05](05_DATABASE_AND_MIGRATIONS.md), [06](06_PROMETHEUS_AND_STRUCTURED_LOGS.md) | Import DB et observabilité locale ; fin MVP système |
| 11 | [13](13_TEST_STRATEGY.md), consolidation [14](14_GITHUB_ACTIONS_CI.md), [16](16_HARDENING_AND_SECURITY.md) | Audit des protections/tests déjà construits |
| 12 | [12](12_GRAFANA_DASHBOARDS.md), [15](15_TRITON_KERNEL_LAB.md) | Analyse Grafana, Kernel Lab qualifié, campagnes publiées |
| 13 | [17](17_DOCUMENTATION_AND_PORTFOLIO.md) | Rapport et portfolio complet selon preuves réelles |
| Option | [11](11_MODEL_INGESTION.md), [15B](15B_VLLM_KERNEL_INTEGRATION_EXPERIMENT.md) | Uniquement après sélection explicite |

Les gates sans DB fonctionnent dès l'ordre 7 ; le MVP système est complet après ordre 10
et validations CPU/GPU correspondantes. La CI et les tests ne sont pas retardés jusque-là.
La qualification kernel ouverte empêche de déclarer le portfolio kernel complet,
sans bloquer les fondations CPU indépendantes.

## Ce qui reste à mesurer

00B doit établir la compatibilité concrète Kaggle/pilote/Python/PyTorch/vLLM/backend,
Triton et compile. Les paramètres YAML ne sont pas des configurations optimales validées.
AWQ/TP peuvent rester bloqués si la session ne les permet pas ; l'indiquer honnêtement.
Les SLO doivent être calibrés par workload avant une candidate, avec artifacts de calibration.
Ne pas transformer des résultats synthétiques CPU en démonstration de performance GPU.
