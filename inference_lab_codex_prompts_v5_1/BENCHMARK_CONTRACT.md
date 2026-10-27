# InferenceLab — Contrat de benchmark et de résultats

Version du contrat : **1.0** — PDD **1.2** — pack **v5** — 17 octobre 2026.

Ce document est normatif. Les prompts l'appliquent sans réinventer ses définitions.
Les statuts de qualification et les exemples sont des spécifications pré-implémentation,
pas des résultats GPU. Aucun couple de versions n'est déclaré validé par ce pack.

## 1. Frontières et modèle de charge

- Runner HTTP dans un processus distinct du serving ; gateway et vLLM sont également distincts.
- Campagnes de référence en loopback dans une seule session Kaggle.
- Modèle fermé `closed_loop` : C workers, une requête active par worker, nouvelle requête
  après terminaison de la précédente, sans think time. C est un plafond côté client,
  pas le nombre de séquences effectivement exécutées par le scheduler.
- Pas d'affirmation de SLO sous arrivées indépendantes à partir de ce modèle fermé.
- Pool HTTP persistant, connexions et limites configurées pour C ; pas de client recréé
  pour chaque requête. Préparer les payloads et tokeniser avant la fenêtre mesurée.
- Pas de retry automatique, ni côté client ni côté gateway.
- Les warmups sont terminés et le serveur drainé avant le début de mesure.
- La campagne est un lot fini de N tentatives : démarrage et vidange inclus dans son débit.
  Ne pas appeler ce débit une capacité stationnaire sans étude complémentaire.

## 2. Streaming et cycle de vie

Les campagnes avec SLO TTFT/TPOT utilisent `stream=true` de bout en bout.
Le proxy transmet les événements SSE sans attendre la réponse complète et sans
retokenisation synchrone. Il préserve les options de génération et l'usage final
quand celui-ci est pris en charge par la version vLLM qualifiée.

Un HTTP 200 n'est pas un succès de génération. Succès = flux valide terminé selon
le contrat API qualifié, sans erreur, avec finish reason attendu et compteurs nécessaires.
Une fin de flux abrupte, un timeout, une erreur SSE ou une déconnexion est un échec.
En cas de déconnexion downstream, fermer la requête upstream et vérifier son arrêt
dans le smoke test GPU. En cas de travail résiduel non drainé, ne pas enchaîner un
point de mesure : le déclarer invalide ou redémarrer proprement la cible.

Définir un timeout global et un timeout d'inactivité de streaming configurables.
Ne jamais transformer un échec après émission des headers en réponse JSON HTTP fictive.

## 3. Horloge et événements

Utiliser `time.perf_counter_ns()` ou une horloge monotone équivalente côté runner.
Les durées d'une requête sont calculées avec la même horloge. UTC sert à corréler
les artifacts, jamais à calculer une latence par différence entre machines.

- `request_start_ns` : immédiatement avant l'appel HTTP, après admission dans le slot.
  Il inclut le travail réseau/HTTP et l'éventuelle attente dans le pool de cet appel.
- `first_content_ns` : réception du premier événement contenant du texte généré non vide.
  Exclure rôle seul, métadonnées, usage et marqueur de terminaison.
- `last_content_ns` : réception du dernier événement de contenu généré non vide.
- `terminal_ns` : fin valide du flux (usage final inclus si attendu) ou instant de l'échec.
- `client_queue_ms` : attente avant admission dans un slot si une implémentation en a une ;
  distincte de la latence HTTP. Dans le modèle C workers, ne pas créer N tâches en attente
  puis présenter leur attente comme une file runtime.

## 4. Métriques canoniques

| Champ | Définition |
| --- | --- |
| `client_ttft_ms` | `(first_content_ns - request_start_ns) / 1e6` |
| `client_e2e_ms` | `(terminal_ns - request_start_ns) / 1e6` pour une réponse réussie |
| `client_tpot_estimate_ms` | `(last_content_ns - first_content_ns) / 1e6 / (output_tokens - 1)` |
| `successful_requests_per_second` | Réponses réussies / durée de la fenêtre |
| `successful_output_tokens_per_second` | Somme des output tokens des réponses réussies / même durée |
| `error_rate` | Tentatives échouées / tentatives mesurées |

La fenêtre d'un run est `[min(request_start_ns), max(terminal_ns)]` sur toutes ses
tentatives, y compris les échecs. Warmup exclu. Une campagne sans tentative ou de durée
nulle est invalide, et ne produit pas des taux artificiels à zéro.

Le TPOT est une **estimation de réception client**, pas une mesure exacte du kernel
decode ni une distribution d'inter-token latency. Un événement SSE peut contenir
plusieurs tokens. Si `output_tokens <= 1`, si moins de deux événements de contenu ont
été observés, ou si les données nécessaires manquent, l'estimation vaut `null` avec
une raison explicite. Ne pas diviser par le nombre de chunks.

Les compteurs d'usage du serveur sont la source de référence des tokens ; conserver
`token_count_source`. Ne pas remplacer silencieusement par une retokenisation du texte.
La qualification vérifie leur sémantique avec le checkpoint et les paramètres choisis.
Les chunks coalescés et les tokens non visibles limitent l'interprétation du TPOT estimé.

Ne pas imposer « prefill + premier decode » comme décomposition interne universelle du
TTFT : le prefill peut déjà produire les logits du premier token.

## 5. Agrégation et valeurs absentes

Les percentiles client portent sur les réponses réussies du même point de mesure,
avec valeur disponible. Utiliser la méthode linéaire : index `h=(n-1)*q`, interpolation
entre les valeurs triées adjacentes (équivalent NumPy `method="linear"`). Enregistrer
la méthode et la population. Ne pas moyenner des p95 pour fabriquer un p95 global.

Champs agrégés : `client_ttft_p50_ms`, `client_ttft_p95_ms`, `client_ttft_p99_ms`,
`client_e2e_p50_ms`, `client_e2e_p95_ms`, `client_e2e_p99_ms`,
`client_tpot_estimate_p50_ms`, `client_tpot_estimate_p95_ms`,
`client_tpot_estimate_p99_ms`, et les trois taux de la section 4.

Chaque métrique inclut valeur, unité, source, nombre d'observations et raison si absente.
Conserver séparément timeouts, erreurs HTTP, erreurs de flux, longueurs inattendues et
réponses partielles. Les tokens des réponses partielles ne gonflent pas le débit réussi.
Les percentiles conditionnels aux succès ne représentent pas les requêtes censurées.
Toujours afficher nombre de tentatives, succès, erreurs et couverture des métriques.

Les métriques `runtime_*` et `gpu_*` restent distinctes des métriques `client_*`.
Les histogrammes Prometheus servent au diagnostic ; les gates client sont recalculés
depuis les mesures brutes du runner. Une métrique absente n'est jamais remplacée par zéro.

## 6. Workload effectif

Le dataset est versionné, non sensible et identifié par nom, version et SHA-256 des
octets. Préparer un plan de requêtes déterministe et en enregistrer le hash : IDs,
ordre, payloads effectifs et paramètres de génération, sans secrets.

Enregistrer `repo_id`, révision résolue immuable, tokenizer/révision, hash du chat
template et paramètres du template. Mesurer les entrées **après** templating.
Fixer `n=1`, sampling, seed, stop et politique EOS. Pour la référence synthétique,
température 0 et sorties de longueur contrôlée ; `ignore_eos=true` uniquement après
qualification du support de cette extension vLLM, transmis aussi par le gateway.
Enregistrer cette politique et les longueurs réellement obtenues. Une longueur
inattendue invalide la conformité du run au workload, sans être masquée comme succès
de benchmark. Le succès de transport peut être enregistré séparément.

Workloads initiaux : short 128/128, long 512/128 puis 2048/128, generation-heavy 128/512
(entrée/sortie cibles). Entrée + sortie doit respecter `max_model_len`. Les tailles
4k/8k nécessitent une configuration qualifiée distincte ; pas d'extension automatique.
La seed du workload ne constitue pas une garantie de déterminisme bit à bit du serving.

## 7. Warmup, caches et runtime effectif

Prefix caching explicitement désactivé pour les campagnes de référence. Ne pas
dépendre du default d'une version. Un test spécifique de cache est hors campagne initiale.
Warmup séparé pour les shapes et concurrences mesurées ; enregistrer son volume et
vérifier readiness, compilation/capture initiales et absence de travail en attente.

Le manifeste de lancement conserve config attendue, config effective, commande sans
secrets, logs de démarrage et origine de chaque valeur. Fixer : dtype FP16,
quantification/backend linéaire effectif, backend d'attention, TP, `max_model_len`,
`max_num_seqs`, `max_num_batched_tokens`, budget mémoire, KV dtype, prefix caching,
chunked prefill et mode CUDA graphs/eager. Capturer les valeurs résolues par le runtime.
Si une valeur critique ne peut être vérifiée, l'indiquer : ne pas la certifier à partir
du YAML attendu ni du seul endpoint `/v1/models`.
Un changement voulu requiert redémarrage documenté et nouveau manifeste ; le MVP
n'implémente pas d'orchestrateur de redéploiement automatique.

## 8. Profils d'exécution et stockage

**Kaggle** : runner, vLLM, collecteur GPU et gateway lorsque testé, dans la même
session ; HTTP loopback. JSON et fichiers bruts autonomes, aucune base distante requise.
Pas d'hypothèse Docker Compose/systemd. Lancement CLI dans un environnement Python isolé.

**Local/CI** : gateway/mock, PostgreSQL, Prometheus et Grafana pour le développement
et l'analyse. PostgreSQL importe les artifacts par `run_id`, de façon idempotente ;
il est un index reconstruit depuis les artifacts, pas leur source de vérité.
Pas d'écriture PostgreSQL synchrone par requête dans le parcours benchmark initial.

Un artifact doit rester exploitable sans PostgreSQL, Prometheus ou Grafana.

## 9. Identités et artifacts

`campaign_id` regroupe des runs. Un `run_id` identifie exactement un workload,
une configuration, une concurrence et une répétition ; aucune métrique client ne
mélange les niveaux de concurrence ou les répétitions.

```text
results/<campaign_id>/<run_id>/
  experiment.yaml
  result.json
  report.md
  environment.json
  runtime_manifest.json
  nvidia-smi.txt
  nvidia-smi-topo.txt
  pip-freeze.txt
  server.log
  raw/requests.jsonl
  raw/gpu.jsonl
```

Conserver le dataset/plan non sensible ou une référence immuable permettant de le
reconstruire. Un fichier GPU vide exige une raison dans le résultat. Un run interrompu
garde ses artifacts et son statut incomplet ; il ne reçoit pas PASS.

`result.json` contient `result_schema_version="1.0"`, `metric_contract_version="1.0"`,
identités, état de complétude, config résolue, environnement, workload effectif,
identité du runtime, cibles, compteurs, métriques, qualité/limites et références brutes.
`experiment.yaml` et les policies ont également `schema_version="1.0"`.

Dans `requests.jsonl`, conserver au minimum `request_id`, `dataset_request_id`,
index du plan, statut de transport et de conformité workload, timestamps de section 3
(nullables), nombre d'événements de contenu, input/output tokens et leur source,
finish reason, erreur normalisée et raison des valeurs manquantes. Ne pas conserver
le contenu des réponses par défaut ni des timestamps token par token fictifs.

L'environnement distingue GPU présents/visibles/utilisés et GPU UUID/index ; GPU count
pour comparabilité = nombre **utilisé**. Capturer pilote, CUDA supportée par le pilote,
runtime CUDA de PyTorch, toolkit s'il existe, Python, OS, CPU/RAM, versions torch/vLLM/
Triton, git SHA et état dirty. Le champ CUDA affiché par `nvidia-smi` ne suffit pas.

## 10. Répétitions et niveau de preuve

- Smoke : petit nombre de requêtes, vérifie le fonctionnement ; pas de certification p99.
- Exploration : une passe possible, résultats descriptifs explicitement identifiés.
- Comparaison publiée avec gate : au moins trois répétitions par configuration,
  mêmes plans de requêtes appariés et ordre alterné baseline/candidate dans une session.
- Calibrer le volume et la dispersion sur des runs baseline indépendants avant de
  figer la policy. Aucun minimum de répétitions ne garantit à lui seul la précision.
- Enregistrer volume minimal par métrique, budget de bruit et protocole dans la policy.
  Si leur justification manque, ne pas convertir le run en PASS de régression.
- p99 descriptif possible sur petit N mais marqué faible support ; 200 requêtes ne
  donnent qu'environ deux observations dans le dernier pourcent.

Les SLO sont empiriques sur l'échantillon. Zéro erreur sur 200 requêtes ne démontre
pas p<1 % à 95 % de confiance (borne unilatérale proche de 1,5 % sous indépendance).
Pas de promesse statistique déduite du simple error_rate observé.

## 11. Comparabilité

Chaque comparaison déclare `kind` (`regression` ou `exploratory`) et une liste explicite
`allowed_changes` de chemins de configuration. Pas de wildcard autorisant tout.
Toute différence critique non autorisée rend la comparaison invalide. Le rapport
énumère les différences observées, y compris celles attendues.

Invariants par défaut : hardware utilisé/topologie et profil de placement, modèle et
révision, tokenizer/template, dataset/plan, sampling/longueurs, cible, contrat de
métrique, concurrence, protocole warmup/mesure, dtype/quantification, TP et runtime.
Le volume mesuré doit satisfaire la même policy ; les répétitions doivent être appariées.
Le changement testé (par exemple `runtime.version`) peut être autorisé dans un gate ;
ne pas exiger simultanément son égalité. Les changements de dépendances réellement
induits sont également déclarés et limitent l'attribution causale.

AWQ : même modèle source vérifié, tokenizer, nombre de GPU et TP. Enregistrer la
provenance du checkpoint quantifié ; ses révisions différentes sont attendues et
explicitement autorisées en exploration. Pas de comparaison 3B FP16/7B AWQ présentée
comme effet de quantification.

TP : même checkpoint/dtype/workload, nombre de GPU et TP explicitement variables.
Expérience exploratoire, pas gate standard. FP16/TP2 versus AWQ/TP1 = comparaison de
configurations de déploiement, ne permettant pas d'isoler AWQ ni TP.

Direct/gateway : même serving effectif et même plan ; ordre alterné. Une différence
entre deux p95 est un écart de percentiles, jamais le p95 d'un overhead individuel.

## 12. Policies et verdicts

Les seuils SLO dépendent du workload et sont figés après calibration, avant la candidate.
Les valeurs 500 ms/40 ms ne sont pas des objectifs T4 acquis. Un template non calibré
est accepté pour démontrer le mécanisme CPU, pas pour certifier une performance GPU.

| Verdict | Condition | Code CLI |
| --- | --- | --- |
| PASS | Données valides, suffisantes et policy satisfaite | 0 |
| FAIL | Violation établie selon la policy | 1 |
| INCONCLUSIVE | Volume/couverture insuffisant ou effet dans la zone de bruit | 2 |
| INVALID | Données incomplètes, schéma incohérent ou différence non autorisée | 2 |

Une métrique requise entièrement absente = INVALID ; couverture/volume trop faibles =
INCONCLUSIVE. Une métrique optionnelle absente n'invalide pas les autres.
Une latence basse avec échecs n'est jamais présentée seule comme succès.

Algorithme minimal d'un gate de régression : comparer chaque paire de répétitions,
calculer l'augmentation relative pour les latences ou la baisse relative pour le débit,
puis donner l'intervalle min/max des changements appariés. Cet intervalle est une
dispersion observée, **pas** un intervalle de confiance.
Pour un seuil de dégradation T et une marge de bruit B calibrée : PASS si tous les
changements <= T-B ; FAIL si tous > T+B ; sinon INCONCLUSIVE. B doit être >=0,
documentée et ne pas rendre le critère de PASS négatif. Baseline nulle/non finie
pour une métrique relative : comparaison invalide, sans division artificielle.

Pour les SLO empiriques, évaluer chaque répétition valide : toutes conformes = PASS,
toutes non conformes = FAIL, mélange = INCONCLUSIVE. Conserver chaque violation brute.
Combiner les clauses : INVALID prioritaire ; sinon FAIL si une clause échoue, sinon
INCONCLUSIVE si une clause ne permet pas de conclure, sinon PASS.

## 13. Contrat YAML unique

Le bloc suivant est un **template de configuration**, pas une campagne GPU qualifiée.
Le loader doit refuser le placeholder de révision pour un run publié. `schema_version`
est une chaîne. Les dtypes canoniques sont `float16`, `float32`, `bfloat16` ; T4
référence = `float16`, sans conversion silencieuse. Aucun exemple T4 n'utilise BF16.
Tous les defaults runtime ci-dessous doivent être qualifiés avant usage publié.

```yaml
schema_version: "1.0"
name: qwen3b-fp16-reference
model:
  repo_id: Qwen/Qwen2.5-3B-Instruct
  revision: "<IMMUTABLE_COMMIT_REQUIRED>"
  tokenizer_repo_id: Qwen/Qwen2.5-3B-Instruct
  tokenizer_revision: "<IMMUTABLE_COMMIT_REQUIRED>"
  trust_remote_code: false
runtime:
  backend: vllm
  dtype: float16
  quantization: null
  tensor_parallel_size: 1
  max_model_len: 4096
  max_num_seqs: 32
  max_num_batched_tokens: 4096
  gpu_memory_utilization: 0.85
  kv_cache_dtype: auto
  enable_prefix_caching: false
  enable_chunked_prefill: true
  enforce_eager: true
target:
  kind: direct
  url: http://127.0.0.1:8001/v1/chat/completions
workload:
  dataset: experiments/datasets/short-chat-v1.jsonl
  load_model: closed_loop
  concurrency: [1, 4, 8, 16, 32]
  seed: 42
generation:
  stream: true
  n: 1
  temperature: 0.0
  max_tokens: 128
  ignore_eos: true
warmup:
  requests_per_point: 20
measurement:
  requests_per_run: 200
  repetitions: 3
  purpose: exploratory
policy_file: null
```

Le budget 0.85, les limites et `enforce_eager` sont des candidats de qualification,
pas des valeurs optimales. Le manifeste enrichit ce YAML avec backend effectif,
version, hash de template/plan et valeurs résolues. Pas de champs fictifs pour prouver
qu'un backend a été réellement utilisé.

Le SLO et la régression résident exclusivement dans un fichier policy versionné,
référencé par `policy_file` (pas une seconde syntaxe inline). Structure à implémenter :
`schema_version`, `policy_id`, `workload_id`, `calibration_ref`, `required_metrics`,
`min_samples_per_metric`, `min_repetitions`, `slo` (champ métrique -> `{max: nombre}`),
`regression` (champ -> `{max_relative_degradation: nombre, noise_margin: nombre}`),
`comparison` (`kind`, `allowed_changes`). Valider les unités, seuils et directions.
Une policy de test utilise uniquement des fixtures identifiées synthétiques.

## 14. Acceptation avant publication

1. Flux direct et gateway testés, annulation et erreurs partielles comprises.
2. Agrégats recalculables à l'identique depuis `raw/requests.jsonl`.
3. Métadonnées qualifiées, workload conforme et limites documentées.
4. Aucune métrique critique fabriquée ; aucune différence silencieuse entre runs.
5. Gate capable de produire les quatre verdicts avec raisons et différences explicites.
6. Les références GPU publiées correspondent à une exécution réelle documentée.


## Convention de nommage — TPOT

Utiliser strictement :

```text
client_tpot_estimate_ms
runtime_tpot_ms
```

`client_tpot_estimate_ms` ne doit jamais être renommé simplement `tpot_ms`.

Un événement SSE n'est pas un token.

## Répétitions

Trois répétitions par configuration sont un **minimum opérationnel**, pas une garantie statistique. Si la dispersion ne permet pas de conclure, le verdict doit pouvoir être `INCONCLUSIVE`.
