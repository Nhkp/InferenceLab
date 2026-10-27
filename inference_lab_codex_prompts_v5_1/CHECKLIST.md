# Checklist d'exécution — pack v5 / PDD v1.2

## Préparation et contrats

- [ ] Lire 00, PDD et BENCHMARK_CONTRACT ; distinguer should/optional des exigences.
- [ ] 00B : qualification serving FP16 streaming ; logs/manifeste/versions réelles.
- [ ] 00B : état exact Triton/compile/AWQ/TP, dont not_tested si non exécuté.
- [ ] 01 + 14 : bootstrap et CI CPU sans GPU/poids.
- [ ] 02 : YAML/policy/résultats canoniques, unités/sources/null et versions.

## MVP système

- [ ] 03 : vLLM direct, fake SSE, readiness réelle et annulation.
- [ ] 08 : modèle fermé, métriques correctes, données brutes recalculables.
- [ ] 09 : run distinct par point/répétition, warmup/cache/config effectifs.
- [ ] 10 : quatre verdicts, volume/bruit et comparabilité/changements autorisés.
- [ ] 04 : gateway progressif sans buffering ni retry, erreurs après 200.
- [ ] 07 + 07B : télémétrie disponible, runbook/export/nettoyage Kaggle.
- [ ] 05 : import PostgreSQL idempotent, aucune dépendance DB de la mesure.
- [ ] 06 : logs/métriques fidèles aux sources et à la fin réelle des streams.
- [ ] Tests/CI/protections présents à chaque étape ; audit 13/14/16.

## Portfolio complet

- [ ] Campagnes contrôlées réelles, répétitions et artifacts publiables.
- [ ] 12 : Grafana d'analyse avec métriques réellement disponibles.
- [ ] 15 : Kernel Lab correct, stack qualifiée, résultats bruts et limites.
- [ ] 17 : README/rapport fidèles à l'état réel, aucune performance inventée.

## Extensions sélectionnées uniquement

- [ ] Décision explicite avant 11, 15B ou workflow GPU self-hosted.
- [ ] Should improve priorisés séparément, sans implémentation automatique.

Après chaque étape : git status, lint, typecheck, tests pertinents et commit des seuls
fichiers utiles (pas de git add . aveugle incluant poids, secrets ou caches).
Un skip GPU ne clôt pas une qualification ; un run interrompu ne reçoit pas PASS.
