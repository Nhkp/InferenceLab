# Prompt 07B — Runbook et export Kaggle, après qualification 00B

Finaliser les scripts du spike, sans reconstruire une stack système complète.
Créer/mettre à jour kaggle/README.md, bootstrap.sh, capture_environment.py et
run_experiment.sh. CLI/Python sources de vérité ; notebook mince facultatif.

bootstrap : environnement isolé, version Python candidate puis qualifiée, installation
verrouillée cohérente avec vLLM, aucune dépendance latest implicite ; vérifier réseau,
ressources et emplacement d'export. Ne pas changer la stack préinstallée globalement
pour contourner silencieusement une incompatibilité. Pas de version GPU inventée.

Le runbook : résoudre modèle/tokenizer en commits, choisir mode GPU, lancer vLLM,
capturer manifeste/logs, attendre readiness avec génération réelle, lancer le runner
sur loopback, exporter après chaque run puis arrêter/nettoyer les processus et vérifier
le drain. Ne pas mesurer pendant téléchargements, compilation initiale ou Kernel Lab.
Pas de DB distante, tunnel, Docker Compose, systemd ou runner GitHub permanent requis.

Single GPU 0/TP1, dual GPU 0,1/TP2 conditionnel. Principal ~3B FP16 ; AWQ même source
et mêmes ressources pour son étude ; TP même checkpoint/dtype. ~7B optionnel.
Capture complète et arborescence par campagne/run du contrat. Préserver aussi runs
interrompus, statut incomplet et artifacts déjà disponibles. Documenter mécanisme
effectif d'export Kaggle ; ne pas supposer la persistance implicite de la session.

Tests CPU : scripts d'environnement tolérants aux packages absents, configuration et
erreurs explicites. Smoke GPU séparé : commandes réellement lancées et statut exact.
