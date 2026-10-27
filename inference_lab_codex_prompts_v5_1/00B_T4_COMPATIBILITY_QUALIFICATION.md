# Prompt 00B — Qualification Kaggle / T4

Qualifie l'environnement GPU de référence avant toute implémentation substantielle.

Lis d'abord le PDD v1.2, `BENCHMARK_CONTRACT.md` et `00_GLOBAL_RULES.md`.

## Objectif

Valider réellement sur Kaggle/T4 :

- vLLM + Qwen ~3B + FP16 ;
- streaming OpenAI-compatible ;
- calculabilité de `client_ttft_ms`, `client_e2e_ms` et `client_tpot_estimate_ms` ;
- `torch.compile` ;
- Triton ;
- TP=2 ;
- AWQ.

Aucun fallback silencieux n'est autorisé.

## Artifacts de qualification

Créer :

```text
qualification/<qualification-id>/
├── environment.json
├── nvidia-smi.txt
├── nvidia-smi-topo.txt
├── pip-freeze.txt
├── server-command.txt
├── server-log.txt
├── stream-sample.txt
├── qualification.json
└── qualification-report.md
```

Utiliser un identifiant du type :

```text
kaggle-t4-YYYYMMDD-a
```

## Étape 1 — Environnement

Capturer :

- Python ;
- OS ;
- GPU names/count ;
- VRAM/GPU ;
- compute capability ;
- NVIDIA driver ;
- CUDA ;
- PyTorch ;
- vLLM ;
- Triton ;
- transformers ;
- git SHA ;
- model/tokenizer revision.

## Étape 2 — 1×T4 FP16

Utiliser explicitement :

```text
CUDA_VISIBLE_DEVICES=0
tensor_parallel_size=1
dtype=float16
```

Choisir un checkpoint Qwen ~3B et pinner une révision immuable.

Valider par une **vraie génération streaming**, pas seulement `/v1/models`.

## Étape 3 — Streaming

Avec une horloge monotone, enregistrer :

- `t0` avant l'appel HTTP ;
- premier contenu généré non vide ;
- dernier contenu généré ;
- fin valide du flux.

Vérifier qu'on peut calculer :

```text
client_ttft_ms
client_e2e_ms
client_tpot_estimate_ms
```

Un événement SSE n'est jamais supposé correspondre à un token.

## Étape 4 — torch.compile

Sur une petite opération représentative :

- vérifier correctness ;
- compiler ;
- exécuter après compilation ;
- donner un statut explicite.

## Étape 5 — Triton

Tester un kernel minimal puis, si possible, un RMSNorm simple.

Résultat obligatoire :

```text
VALIDATED | FAILED | UNAVAILABLE | NOT_TESTED
```

Si Triton n'est pas viable sur la T4 :

- conserver l'erreur et les versions ;
- ne pas transformer un skip en succès ;
- garder le serving sur T4 ;
- autoriser plus tard le Kernel Lab sur un GPU compatible distinctement documenté.

## Étape 6 — TP=2

Avec le même checkpoint/dtype si possible :

```text
CUDA_VISIBLE_DEVICES=0,1
tensor_parallel_size=2
dtype=float16
```

Valider seulement le démarrage et une génération streaming. Ce n'est pas encore un benchmark.

## Étape 7 — AWQ

Qualifier une variante AWQ liée au même modèle source lorsque possible :

- chargement ;
- génération ;
- streaming ;
- chemin de quantification observable.

Ne tirer aucune conclusion de performance.

## Étape 8 — Manifeste final

Créer `qualification.json` avec :

```json
{
  "qualification_id": "...",
  "single_gpu_fp16_vllm": "VALIDATED",
  "streaming": "VALIDATED",
  "torch_compile": "VALIDATED",
  "triton": "VALIDATED",
  "tp2": "VALIDATED",
  "awq": "VALIDATED",
  "validated_versions": {},
  "model": {
    "repo_id": "...",
    "revision": "..."
  }
}
```

Toute capacité non validée garde son vrai statut et sa raison.

## Définition de terminé

Cette étape est finie lorsque :

- une génération FP16 streaming fonctionne sur 1×T4 ;
- le checkpoint et la révision sont fixés ;
- les versions compatibles sont connues ;
- `torch.compile`, Triton, TP=2 et AWQ ont chacun un statut explicite ;
- les artifacts de qualification sont conservés.
