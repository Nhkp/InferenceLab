# InferenceLab — Changelog du pack v5

**Date :** 17 octobre 2026  
**Entrée :** pack v4 Kaggle/T4 et PDD v1.1 joints à la conversation  
**Sortie :** pack v5, PDD v1.2 et contrat de benchmark v1.0.

Cette révision est une correction documentaire avant implémentation. Elle ne contient
pas de plateforme implémentée, de lock GPU prétendument validé ou de résultats Kaggle.
Les documents should et optional sont ajoutés pour décision ultérieure, pas appliqués
automatiquement comme nouveaux travaux.

## Traçabilité des Must fix

| ID | Correction | Fichiers principaux |
| --- | --- | --- |
| M1 | Qualification T4/version/backend avant claims ; aucune compatibilité Triton supposée | 00, nouveau 00B, PDD §6/16, 01, 03, 15 |
| M2 | Streaming end-to-end, usage, annulation et erreurs après HTTP 200 | Contrat §2, 03, 04, 08, 13 |
| M3 | Définitions, horloge, populations, fenêtre et métriques estimées/absentes | Contrat §3–5, 02, 06, 08, 10 |
| M4 | Modèle fermé explicite, C workers, pool, absence de retries | Contrat §1, 08, 09 |
| M5 | Longueurs effectives, tokenizer/template, EOS, plan et conformité | Contrat §6, PDD §8, 02, 08 |
| M6 | Prefix cache désactivé, warmup/drain et manifeste effectif | Contrat §7, 03, 07B, 09 |
| M7 | Modèle principal ~3B, AWQ mêmes ressources, TP même checkpoint/dtype | Contrat §11, PDD §7/11, 07B, 09, 10 |
| M8 | Smoke/exploration/gate distincts, répétitions/bruit, quatre verdicts | Contrat §10/12, PDD §9/14, 09, 10 |
| M9 | Profils local/Kaggle fermés, loopback, artifacts sans DB et import local | Contrat §8, PDD §4/5/13, 05, 07B, 09 |
| M10 | Schémas canoniques, données brutes, identités et différences autorisées | Contrat §9/11/13, 02, 05, 09, 10 |

## Cohérence transversale

- Les deux copies du PDD d'entrée étaient identiques. Le PDD de l'archive devient la
  référence v1.2 ; le document v1.1 joint séparément n'est pas modifié par ce livrable.
- Remplacement des exemples BF16 T4 et des anciennes variantes YAML par un contrat unique.
- PostgreSQL est un index d'import, jamais une dépendance de mesure sur Kaggle.
- Numéros historiques conservés, ordre d'exécution corrigé dans README et CHECKLIST.
- Grafana après MVP système ; Kernel Lab requis pour portfolio complet ; intégration
  vLLM et registry automatique explicitement conditionnels.
- Tests/CI CPU/protections remontés dans les étapes concernées, audit final conservé.
- Commande de metadata unifiée sous `inference_lab.telemetry.gpu`.
- Pas de nouveau runtime, service cloud, matériel ou fonctionnalité produit ajouté.

## Documents ajoutés

- BENCHMARK_CONTRACT.md : spécification normative commune.
- 00B_T4_COMPATIBILITY_QUALIFICATION.md : spike préalable borné.
- SHOULD_IMPROVE.md : six fiches problème/impact/modification/acceptation.
- OPTIONAL_STRETCH.md : options, préconditions, validation et critères d'arrêt.
- CHANGELOG_V5.md : traçabilité et limites de cette révision.

## Sources de qualification à revalider

Les constats techniques proviennent de la revue documentaire du 17 octobre 2026.
Au moment d'implémenter, consulter les sources de la version choisie et conserver
leurs références dans docs/compatibility.md ; ces pages peuvent changer.

- [Installation GPU vLLM](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
- [Support et installation Triton](https://github.com/triton-lang/triton#compatibility)
- [Support FlashAttention CUDA](https://github.com/Dao-AILab/flash-attention#nvidia-cuda-support)
- [Quantification vLLM](https://docs.vllm.ai/en/stable/features/quantization/)
- [Benchmark vLLM](https://docs.vllm.ai/en/stable/cli/bench/serve/)
- [Métriques vLLM](https://docs.vllm.ai/en/stable/usage/metrics/)
- [Prefix caching vLLM](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)
- [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks)

## Vérifications du livrable documentaire

La préparation de l'archive vérifie noms/liens internes, délimitation Markdown,
syntaxe du YAML canonique, absence de clés YAML dupliquées, taille/contenu des membres
et intégrité ZIP. Elle ne remplace ni tests du futur code, ni qualification GPU,
ni calibration de SLO. Le pack ne doit pas être décrit comme validé sur Kaggle.
