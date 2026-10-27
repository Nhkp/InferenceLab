# Prompt 10 — SLO et gates avec quatre verdicts

Lire BENCHMARK_CONTRACT §10–13 ; appliquer son algorithme sans conventions supplémentaires.
Créer policy versionnée unique, chargée par policy_file ou argument CLI explicite,
et ComparisonResult contenant différences, clauses, volumes, bruit et verdict.
CLI : `python -m inference_lab.benchmark.policy --baseline BASELINE_CAMPAIGN
--candidate CANDIDATE_CAMPAIGN --policy experiments/policies/example.yaml`.

Valider d'abord schéma/intégrité/complétude, correspondance des points et métriques
requises. Comparaison regression/exploratory et allowed_changes explicites. Refuser
les différences non déclarées ; autoriser runtime.version quand c'est la variable du
gate, et déclarer les dépendances qui changent avec elle. Vérifier GPU utilisés,
topologie, modèle/provenance, template, plan, longueurs, cible, protocole et contrat.
TP1/TP2 et AWQ sont exploratoires ; pas de gate standard les assimilant à des invariants.

SLO calibrés par workload puis figés avant candidate ; aucune valeur GPU acquise dans
les fixtures. Au moins trois paires pour gate publié, volume minimal/marge de bruit
justifiés. Mettre les résultats insuffisants dans INCONCLUSIVE, pas PASS. Absence d'une
métrique requise ou différence critique non autorisée = INVALID.
Calcul des dégradations appariées et marge de bruit exactement selon le contrat ;
min/max observé n'est pas un intervalle de confiance. SLO empiriques ≠ garantie statistique.

Codes : 0 PASS, 1 FAIL, 2 INCONCLUSIVE ou INVALID, avec raison structurée distincte.
Tests : quatre verdicts, mixed repetitions, limite/bruit, valeur nulle/non finie en
baseline relative, métrique absente, petit N, changement autorisé/non autorisé,
hardware utilisé et version de schéma incompatible. Toutes fixtures synthétiques.
