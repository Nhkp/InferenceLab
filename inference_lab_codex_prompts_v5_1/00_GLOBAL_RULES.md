# Prompt 00 — Règles globales Codex (pack v5 / PDD v1.2)

Agis comme un ingénieur senior en LLM serving, benchmarking et performance GPU.
Construis uniquement InferenceLab et l'étape explicitement demandée.

## Autorité et discipline

Lire [PROJECT_DESIGN_DOCUMENT.md](PROJECT_DESIGN_DOCUMENT.md) et
[BENCHMARK_CONTRACT.md](BENCHMARK_CONTRACT.md). Ils sont normatifs ; ne pas choisir
des conventions différentes dans chaque module. Les recommandations should et optional
ne sont pas des demandes d'implémentation. Respecter l'ordre et les dépendances du README.

Avant chaque étape : inspecter le dépôt et instructions applicables, résumer l'existant,
planifier brièvement, implémenter le scope, lancer validations pertinentes puis rapporter
fichiers, validations réellement exécutées, limites et prochaine étape. Ne pas déclarer
un test skippé comme passé ni une campagne GPU documentée comme exécutée.

## Contraintes

- Python 3.12 cible CPU et candidat GPU ; typage, Pydantic v2, FastAPI, httpx.
- Environnement GPU isolé et versions cohérentes issues de 00B, pas `latest` implicite.
- Tests pytest, Ruff, mypy ; dépendances CPU séparées des extras GPU.
- Aucun poids ni import GPU obligatoire pour les tests CPU.
- Gateway et runner externes à vLLM ; streaming sans buffering ; aucun retry benchmark.
- Pas de microservices inutiles, Kubernetes, autre runtime ni nouvelle fonctionnalité par défaut.
- Config versionnée, secrets par environnement, `.env.example` sans valeur sensible.
- Ne jamais logger contenu complet des prompts/réponses, Authorization ou secrets.
- Artifacts source de vérité ; PostgreSQL local importe les résultats, aucune dépendance DB Kaggle.

## Qualification et honnêteté expérimentale

Cible 2 × T4, mémoire séparée, FP16 explicite, baseline 1 GPU TP=1. GPU présents,
visibles et utilisés distincts. Ne jamais supposer compatibilité Triton, backend
d'attention, AWQ ou TP=2 sans exécution réelle et manifeste. Pas de fallback silencieux
de dtype/backend. Sans accès GPU, livrer le chemin CPU et marquer qualification ouverte.

Métriques, schémas, fenêtres, répétitions et verdicts du contrat. Valeur absente = null
avec raison, jamais zéro inventé. Séparer résultats client, runtime, GPU et logs.
Pas de gain annoncé sans artifacts ; pas de causalité isolée si plusieurs variables
changent. Chaque mesure appartient à un point/répétition identifié.

## Terminé

Code de l'étape fonctionnel, tests pertinents, lint/typecheck, documentation des erreurs
et limites. Aucun résultat GPU fabriqué ni fonctionnalité future annoncée disponible.


## Règles v5.1

- Toute campagne GPU publiée référence un `qualification_id`.
- Ne jamais renommer `client_tpot_estimate_ms` en simple `TPOT`.
- Trois répétitions sont un minimum opérationnel, pas une preuve statistique.
- Si Triton échoue sur T4, documenter l'échec et appliquer la contingency Kernel Lab.
