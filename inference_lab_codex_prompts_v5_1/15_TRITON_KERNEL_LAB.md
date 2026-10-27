# Prompt 15 — Kernel Lab : RMSNorm en PyTorch et Triton

Crée un laboratoire GPU isolé du serving afin d'étudier un niveau inférieur de la stack d'inférence.

## Précondition bloquante pour les claims GPU

Appliquer 00B et le contrat. Le hardware cible est T4 SM 7.5 ; le support actuel
annoncé par Triton à partir de SM 8.0 impose une exécution de qualification, pas une
simple installation. Utiliser les versions réellement validées, éventuellement dans
un environnement Kernel Lab distinct et verrouillé. torch.compile doit être exécuté
et vérifié ; pas de fallback eager silencieux. Sans qualification, marquer not_tested
ou blocked, ne pas déclarer le volet kernel terminé grâce à un skip.

Le Kernel Lab est requis pour le portfolio complet, après le MVP système. Son échec
ne doit pas casser les modules CPU/serving indépendants ; il reste un blocage explicite
de la promesse kernel, pas une suppression automatique du scope.

## Pourquoi séparé

Le MVP mesure le système complet. Le Kernel Lab permet ensuite de répondre à une autre question : **comment une primitive GPU isolée se comporte-t-elle, et quelles contraintes expliquent sa performance ?**

Ne modifie pas vLLM dans cette étape.

## Primitive

Commencer par RMSNorm, car elle est simple à valider et existe comme opération de modèle dans les stacks LLM.

```text
kernel_lab/rmsnorm/
├── torch_impl.py
├── compiled_impl.py
├── triton_impl.py
├── benchmark.py
├── test_correctness.py
└── README.md
```

## Comparaison

- PyTorch eager
- `torch.compile` si disponible
- Triton

## Correctness

Plusieurs batch/sequence flattenings, hidden sizes, dtypes et seeds. Référence PyTorch + tolérances explicites.

## Performance

Mesurer latency µs, effective bandwidth si bien définie, et speedup. Warmup séparé. Synchronisation CUDA correcte. Metadata GPU/CUDA/PyTorch/Triton.

## Profiling

Ajouter une procédure Nsight Systems/Compute ou PyTorch profiler documentée, sans rendre l'outil obligatoire en CI.

## Résultats

JSON + Markdown. Aucun gain affirmé avant mesure.

## Suite

À la fin, documenter si ce kernel est un candidat raisonnable à un essai d'intégration vLLM. Ne pas prétendre qu'il remplace déjà un kernel vLLM.


## Hardware de référence du Kernel Lab

Les résultats officiels du Kernel Lab ciblent une **NVIDIA Tesla T4**, uniquement après qualification réelle de la stack.

Chaque benchmark doit enregistrer :

```text
GPU = Tesla T4
compute capability
CUDA
PyTorch
Triton
shape
dtype
warmup
iterations
```

Commencer en FP16.

Tester plusieurs `hidden_size` et nombres de lignes/tokens afin de montrer que le speedup dépend de la shape.

Ne pas publier de tableau de performance sans l'artifact brut correspondant.
