# Prompt 15B — Stretch : intégrer expérimentalement un kernel au chemin vLLM

Cette étape est optionnelle et ne doit être tentée que lorsque le Kernel Lab est correct et benchmarké.

## Objectif

Tester si un kernel custom peut être invoqué dans un chemin d'exécution vLLM réel, sans prétendre construire un backend complet.

## Première piste : Triton

1. Inspecter la version vLLM réellement installée.
2. Identifier précisément l'opération RMSNorm utilisée par le modèle choisi.
3. Vérifier l'API `CustomOp` / `register_oot` disponible dans cette version.
4. Construire un remplacement out-of-tree minimal si l'API le permet proprement.
5. Le remplacement peut appeler le kernel Triton du Kernel Lab si les signatures, strides, dtype, device et semantics sont compatibles.
6. Ajouter un flag d'activation explicite et une voie de retour au comportement vLLM standard.

Ne pas patcher/forker vLLM dans cette extension de référence ; documenter le blocage.

## Validation

Trois niveaux obligatoires :

- correctness de l'opération ;
- smoke test d'une génération réelle ;
- benchmark end-to-end baseline vs custom kernel.

Un kernel isolé plus rapide peut rendre le système identique ou plus lent. Le rapport doit accepter ce résultat.



Produire d'abord une note technique qui détermine le pont nécessaire entre :

- tenseurs PyTorch/vLLM ;
- pointeurs mémoire CUDA ;
- stream CUDA courant ;
- lancement du kernel ;
- lifetime / synchronization ;
- autograd inutile en inference.

Seulement si un pont sans copie et respectant le current CUDA stream est démontré,
créer un prototype minimal via le point d'extension effectivement disponible. Un kernel
Triton sur tenseur PyTorch compatible ne nécessite pas systématiquement de binding
natif ; ne pas en ajouter par défaut. Si un tel binding ou un fork est nécessaire,
documenter la limite et arrêter cette extension plutôt que d'élargir le scope.
Inspecter la sémantique réelle de l'opération (dont éventuelle fusion residual/RMSNorm)
avant toute substitution ; un smoke test ne remplace pas la validation numérique.

## Critère de succès

Le succès n'est pas "être plus rapide". Le succès est de mesurer rigoureusement :

- intégrabilité ;
- correctness ;
- overhead d'interop ;
- impact end-to-end ;
- coût de maintenance.
