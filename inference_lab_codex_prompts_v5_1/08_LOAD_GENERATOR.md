# Prompt 08 — Runner direct et instrument de mesure canonique

Implémenter avant le gateway et la DB. Lire intégralement BENCHMARK_CONTRACT.
Client HTTP async externe : cible vLLM direct puis gateway interchangeable.
Entrée : ExperimentConfig canonique ; aucune syntaxe CLI parallèle contradictoire.
Commande : `python -m inference_lab.benchmark.workload --experiment experiments/example.yaml`.

Construire C workers fermés sans think time, pool partagé dimensionné, N tentatives
exactes, payloads prêts avant mesure. Pas de retry. Warmup terminé puis drain avant
fenêtre ; horloge monotone ; état terminal y compris timeout/erreur après 200.
Parser SSE incrémental, contenu non vide, usage final, annulation et fragments réseau.
Un chunk n'est pas un token. Appliquer exactement TTFT/E2E/TPOT estimé, null/raisons,
fenêtre commune et populations/percentiles du contrat ; aucun débit calculé par somme
de débits individuels. Exclure tokens des réponses échouées du débit réussi.

Préparer datasets versionnés short/long/generation-heavy et plan déterministe. Fixer
tokenizer/template, longueur après template, sampling/EOS et sortie cible ; conserver
longueurs effectives et conformité. Arrêter la publication d'un workload non conforme.
Runtime/cache explicites et vérifiés, pas de cache de préfixe implicite du warmup.

Produire par point/répétition result.json + report.md + raw/requests.jsonl, identités
et metadata du contrat. Dépendance DB interdite ; aucune génération de résultats factices
présentés GPU. Les fixtures CPU sont identifiées synthétiques.

Tests déterministes : événements rôle/vides/usage, coalescence multi-token, fragmentation,
une seule trame de contenu (TPOT null), N<=1, timeout/fin abrupte, annulation, erreurs,
durée incluant échecs/vidange, percentiles et recalcul des agrégats depuis raw.
Le faux serveur prouve l'incrémentalité et le plafond C sans dépendre de performances GPU.
