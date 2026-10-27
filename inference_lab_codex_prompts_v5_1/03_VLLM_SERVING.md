# Prompt 03 — Serving vLLM direct, streaming qualifié

Appliquer le contrat ; utiliser les versions/chemins validés par 00B. vLLM reste
un processus externe à FastAPI. Première cible du runner : vLLM direct en loopback.

Préparer commandes CLI Kaggle sans Docker et profil Compose GPU pour hôte compatible.
Configuration : checkpoint/révisions immuables, FP16, quantification optionnelle,
CUDA_VISIBLE_DEVICES, TP, contexte, scheduler, budget mémoire, KV dtype, cache,
chunked prefill et graphs/eager. Aucun fallback de précision/backend masqué.
Single : GPU 0, TP=1. Dual : GPU 0,1 et TP=2 seulement après qualification.

Créer client HTTP async réutilisable pour POST /v1/chat/completions, streaming SSE,
timeouts global/inactivité, réseau, erreurs HTTP, événements d'erreur et fin abrupte.
Usage final transmis quand supporté ; annulation ferme upstream. Ne pas charger de modèle.

Produire runtime_manifest.json : configuration attendue/effective et preuve/origine,
commande expurgée, version/backend réellement choisis, logs et identités GPU utilisés.
/v1/models seul n'atteste pas dtype/TP/scheduler. Les valeurs invérifiables restent signalées.
Readiness : serveur prêt + génération réelle de smoke hors mesure, pas seulement port ouvert.

Tests CPU avec serveur SSE contrôlé : rôle seul, fragments, multi-token par événement,
usage final, fermeture prématurée, erreur après 200, timeout et annulation.
GPU smoke séparé, aucune dépendance GPU pour tests CPU ni téléchargement automatique.
