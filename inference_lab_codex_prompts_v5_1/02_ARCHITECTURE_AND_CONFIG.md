# Prompt 02 — Schémas canoniques avant runner et stockage

Appliquer le PDD et BENCHMARK_CONTRACT, notamment §9, §11–13. Ne pas inventer une
syntaxe YAML alternative ni conserver les exemples BF16 des packs antérieurs.

Créer Settings : APP_ENV, LOG_LEVEL, HOST, PORT, VLLM_BASE_URL, VLLM_MODEL,
VLLM_TIMEOUT_SECONDS, timeout d'inactivité stream, DATABASE_URL optionnel pour benchmark,
PROMETHEUS_ENABLED, STORE_PROMPT_CONTENT=false. Aucun secret enregistré en résultat.

Pydantic : ModelConfig, RuntimeConfig, TargetConfig, WorkloadConfig, GenerationConfig,
WarmupConfig, MeasurementConfig, ExperimentConfig, ExperimentRun, ExperimentResult,
RequestMeasurement, RegressionPolicy et ComparisonResult. Reprendre exactement les
champs canoniques du contrat. Séparer campaign_id, run_id et répétition/point.
Révision résolue immuable pour résultat publié ; une référence mobile de découverte
ne doit pas être stockée comme preuve de la révision exécutée.

Les valeurs runtime d'exemple sont à qualifier. Vérifier entrée+sortie, streaming pour
SLO client, TP/GPU utilisés, valeurs numériques finies, null et raisons des métriques
absentes. Pas de defaults métriques à zéro. Une policy vit dans policy_file, sans SLO inline.
Définir schémas versionnés, unités, source et population ; incompatibilité de schéma explicite.

Documenter architecture et profils local/Kaggle : artifacts autonomes, DB par import,
gateway distinct, runner externe. Préparer un YAML d'exemple unique conforme au contrat
et des fixtures synthétiques identifiées, sans claim GPU ni calibration inventée.

Tests : parsing/roundtrip, valeurs invalides, placeholders refusés en publication,
versions inconnues, absent/null, verdicts et dimensions de comparaison.
