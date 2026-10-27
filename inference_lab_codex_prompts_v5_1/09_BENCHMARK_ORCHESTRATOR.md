# Prompt 09 — Campagnes séquentielles et artifacts autonomes

CLI : `python -m inference_lab.benchmark.runner experiments/example.yaml`.
Appliquer le contrat, réutiliser le runner existant, ne pas réimplémenter les métriques.

Valider YAML/schema/policy, plan de requêtes et manifeste runtime ; vérifier cible et
config effective. Séparer campaign_id et run_id (un workload, concurrence, répétition).
Warmup par point, mesure, données brutes, agrégats, export puis point suivant.
Campagnes publiées avec gate : répétitions appariées et ordre baseline/candidate
alterné ; volume/bruit issus de calibration distincte et conservée. Campagnes purement
exploratoires identifiées comme telles ; N=200 n'est pas une certification p99.

Matrice limitée, pas de produit cartésien automatique. Variables des expériences du
PDD ; AWQ à ressources identiques ; TP même checkpoint/dtype, exploration conditionnelle.
Comparer chemin direct/gateway sans confondre écart de percentiles et overhead individuel.

Le MVP ne redéploie pas vLLM automatiquement. Décrire les étapes manuelles reproductibles
de redémarrage et recapture du manifeste. Refuser un écart critique entre config attendue
et active : corriger la configuration et recapturer avant de mesurer. Les changements
autorisés concernent la comparaison entre runs, pas un écart attendu/effectif caché.
Aucun accès DB requis pour mesurer.
Ne pas capturer le hardware du laptop comme celui d'un serveur distant : campagne
référence co-localisée, autre cas explicitement documenté et non assimilé.

Conserver erreurs, statut incomplet et artifacts en cas d'interruption ; drain avant
réutilisation de la cible. Report humain : résultats, couverture, différences, limites.
Tests CPU : mini campagne SSE, identité de chaque point, interruption, config mismatch,
workload invalide et génération des artifacts sans PostgreSQL/GPU.
