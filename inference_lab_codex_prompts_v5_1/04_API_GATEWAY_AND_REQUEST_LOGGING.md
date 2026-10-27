# Prompt 04 — Gateway SSE mince, après runner direct fonctionnel

Appliquer le contrat. FastAPI ne charge jamais le modèle.
POST /v1/chat/completions : sous-ensemble documenté, validation, request ID, client
HTTP partagé, forwarding streaming sans buffering, timeout global/inactivité, erreurs,
metrics/logs et annulation upstream sur déconnexion downstream. Aucun retry automatique.
Préserver paramètres de génération, options EOS qualifiées et usage final.

Traiter erreurs après headers comme erreurs de stream, pas JSON 500 rétrospectif.
Mesurer fin de streaming et non le simple retour d'un objet StreamingResponse.
Logs : request_id, run_id, modèle/backend, statut terminal, token counts/source,
durées réellement observées et erreur normalisée. Aucun contenu/secret par défaut.
Pas de persistance PostgreSQL par requête sur ce chemin benchmark.

La différence entre p95 direct et gateway est un écart de distributions. Ne pas créer
gateway_overhead_ms en soustrayant des percentiles. Toute durée locale instrumentée
doit nommer ses frontières et ne pas être présentée comme overhead causal complet.

Tests de streaming progressif (avant fin upstream), fragmentation, usage, annulation,
erreurs partielles, redaction, validation et isolation du client HTTP.
Comparer les deux cibles avec le même runner/plan, dans la même session, ordre alterné.
