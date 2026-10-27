# InferenceLab

> Reproducible GPU LLM serving experiments, measurements and regression checks.

Template à adapter à l'état réel : remplacer les placeholders, marquer les chemins
non exécutés comme `not_tested`, et les fonctionnalités futures comme roadmap.

## Objectif

Caractériser vLLM sur un hardware donné : concurrence, contexte, quantification,
compromis TP et chemin HTTP ; étudier RMSNorm séparément. Montrer des mesures et
leurs limites, pas seulement une API autour d'un modèle.

## Environnement effectivement validé

Compléter depuis artifacts : GPU utilisés, pilote, versions CUDA distinctes, Python,
PyTorch, vLLM, backend d'attention/quantifié, Triton et torch.compile exécutés.
Cible : Kaggle 2 × T4 SM 7.5, baseline un GPU FP16/TP1. Mémoire séparée.
Table de compatibilité : validé / non supporté / échec / non testé, avec preuve.

## Architecture et profils

Runner HTTP externe -> vLLM direct ou gateway SSE -> vLLM -> GPU.
Kaggle : processus dans la même session, loopback, collecte légère, artifacts sans DB.
Local : mock, gateway, import PostgreSQL, Prometheus/Grafana pour hôte accessible.
Artifacts canoniques ; DB reconstruite par import idempotent ; Kernel Lab indépendant.
Ajouter un diagramme fidèle à l'exécution réelle, sans accès distant supposé.

## Contrat de benchmark

Résumer et lier le contrat normatif : charge fermée, concurrence, streaming, TTFT
client, TPOT estimé (chunk != token), E2E, fenêtre commune, erreurs, coverage,
warmup/caches, longueurs effectives, répétitions et quatre verdicts.
Présenter modèle ~3B principal, AWQ à ressources identiques et TP même checkpoint/dtype.
~7B reste optionnel ; ne pas attribuer causalement un résultat à AWQ si TP change aussi.

## Quickstart réellement testé

Insérer commandes CPU/mock, tests/CI, qualification Kaggle, démarrage vLLM, campagne
directe/gateway, export, import PostgreSQL et comparaison. Pas de commandes placeholders
annoncées comme vérifiées. Aucun téléchargement de poids requis pour les tests CPU.
Expliquer absence de Docker/systemd requis sur Kaggle et gestion des processus.

## Résultats et policies

Exemple d'artifact réel avec campagne/run/point/répétition, versions de schéma,
configuration effective, dataset/plan/template hashes, raw et limites.
Montrer recomputation, différences autorisées et verdicts PASS/FAIL/INCONCLUSIVE/INVALID.
SLO empiriques calibrés par workload ; smoke ne certifie ni p99 ni taux d'erreur réel.
Écart de p95 direct/gateway != p95 d'un overhead individuel.

## Observabilité

Logs, métriques client/runtime/GPU et résultats persistés distincts. Panels uniquement
pour métriques confirmées par version/source ; limites des compteurs et valeurs absentes.
Ne pas substituer un histogramme runtime au percentile client du benchmark.

## Kernel Lab

Décrire qualification T4, RMSNorm, référence/correctness, comparaison eager/compile/Triton,
shapes, protocole GPU, données brutes et limites. Ne pas déclarer compile/Triton validés
par un import. Kernel Lab requis au portfolio complet, indépendant du MVP système.
Un speedup microbenchmark n'est pas un gain serving ; intégration vLLM optionnelle
avec correctness, smoke génération puis comparaison end-to-end si réellement réalisée.
Lier kernel_lab/README.md et conserver séparément tout environnement logiciel distinct.

## Tests, sécurité et limites

Commandes effectivement exécutées, passes/skips distincts, tests streaming/annulation,
aucun secret/contenu loggé par défaut, limites/timeouts et workload conforme.
Contraintes Kaggle/session/stockage/topologie et portée des conclusions explicites.

## État / roadmap

MVP système, portfolio complet et extensions sélectionnées distincts. Should et optional
ne deviennent pas des features existantes par copie du template. Ne pas ajouter un
second runtime, frontend ou plateforme d'orchestration à la roadmap de référence.
