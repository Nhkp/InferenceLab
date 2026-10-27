# Prompt 17 — README final, rapport P15 et portfolio

Transforme le dépôt en projet démontrable.

## README

Pars de `PROJECT_README_TEMPLATE.md` et remplace les placeholders par l'état réel du dépôt.

Il doit expliquer : problème, architecture en trois plans, `Experiment`, quickstart CPU/mock, quickstart GPU/vLLM, benchmark direct vs gateway, SLO/regression gates, observabilité, data model, Kernel Lab, tests, sécurité, limites et roadmap.

Ne présente jamais une feature non implémentée comme terminée.

## Architecture

Diagramme Mermaid avec : config de révision -> runner externe -> gateway optionnel/vLLM -> GPU ; artifacts
canoniques -> import PostgreSQL ; observabilité disponible ; Kernel Lab séparé.
Le diagramme doit distinguer profils local et Kaggle, sans accès distant fictif.

## Rapport technique P15

Créer `docs/project_report.md` :

- contexte et besoin ;
- audit des solutions ;
- architecture cible ;
- industrialisation ;
- coûts et risques ;
- KPI/SLO ;
- contrôle et suivi ;
- résultats ;
- perspectives.

Mettre l'accent sur les compromis : latency/throughput/VRAM, gateway overhead, quantization, reproductibilité et coût GPU.

## ADR

Au minimum : vLLM comme serveur, thin gateway, benchmark runner externe, séparation logs/metrics/results, Docker Compose pour MVP, Prometheus/Grafana, PostgreSQL, Kernel Lab isolé.

## Interview notes

Pitch 30 s, pitch 2 min, questions techniques, démonstration de métriques et résultats réels uniquement.

## Exigence spécifique — documentation du Kernel Lab

Le README principal doit contenir une section substantielle `Kernel Lab` qui ne soit pas réduite à une ligne de roadmap.

Cette section doit expliquer :

- pourquoi le laboratoire kernel fait partie d'InferenceLab ;
- la séparation entre performance système et performance kernel ;
- le choix de RMSNorm comme première primitive ;
- la comparaison `PyTorch eager` / `torch.compile` / `Triton` ;
- la méthodologie de correctness ;
- la méthodologie de benchmark ;
- pourquoi un gain microbenchmark ne prouve pas un gain end-to-end ;
- les trois niveaux : microbenchmark, intégration runtime optionnelle, benchmark serving ;
- les métriques utilisées à chaque niveau ;
- les limites et non-objectifs.

Ajouter également un lien clair depuis le README principal vers `kernel_lab/README.md`.

La première version du projet reste volontairement centrée sur Triton pour les kernels custom.


## Documentation Kaggle / T4

Le README final doit préciser clairement :

- hardware de référence : `2 × NVIDIA Tesla T4, 16 GB / GPU` ;
- FP16 comme baseline ;
- pourquoi le développement commence sur 1 GPU ;
- pourquoi TP=2 est une expérience et non une hypothèse de gain ;
- comment lancer un benchmark depuis Kaggle ;
- comment sont exportés les artifacts ;
- limites liées à l'environnement Kaggle ;
- différence entre environnement local/CI et laboratoire GPU.

Ajouter dans la partie résultats une expérience `1 × T4 vs 2 × T4` si elle a réellement été exécutée. Ne jamais inventer de résultats.

## Contrats v1.2 à respecter

Appliquer PDD/contrat : streaming obligatoire pour TTFT/TPOT estimé, modèle fermé,
répétitions/qualité des mesures, quatre verdicts, qualification réelle et versions
immuables. Documenter toute qualification ouverte ; pas de résultat GPU inventé.
~3B est le modèle principal des études contrôlées AWQ/TP ; ~7B optionnel. SLO calibrés
par workload, pas des constantes universelles. Démontrer le recalcul depuis raw et
l'exécution sans DB, puis l'import idempotent local.
Kernel Lab requis pour portfolio complet, intégration vLLM optionnelle ; Grafana après
MVP système. SHOULD_IMPROVE/OPTIONAL_STRETCH sont des décisions à sélectionner,
pas des fonctionnalités automatiquement disponibles.
