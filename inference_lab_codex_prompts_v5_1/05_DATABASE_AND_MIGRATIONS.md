# Prompt 05 — PostgreSQL comme index des artifacts

Après stabilisation des schémas/artifacts, ajouter SQLAlchemy 2 et Alembic.
La campagne Kaggle fonctionne sans DATABASE_URL ; JSON et raw restent canoniques.
La DB locale n'est ni une dépendance du runner ni une étape par requête du gateway.

Tables : experiment_runs (run_id unique, campaign_id, point/répétition, config et
environnement, versions de schéma, statut et résumé), benchmark_measurements
(métrique, valeur nullable, unité, source, population/percentile et raison absente),
inference_requests (metadata importées utiles, sans contenu), model_revisions
(préparée pour découverte ultérieure). Pas de séries Prometheus, logs bruts massifs
ou traces token par token. Index utiles, pas de modèle EAV inutilement sophistiqué.

Créer import CLI d'un dossier résultat : validation de schéma/intégrité d'abord,
transaction par run, idempotence par run_id et hash du contenu canonique. Même id
avec contenu différent = conflit explicite, pas overwrite silencieux. Import d'un
run incomplet possible avec son statut ; ne pas lui attribuer PASS.
Une DB indisponible signale échec d'import sans corrompre l'artifact ni la réponse LLM.

Tests : migrations, import répété sans doublon, conflit, rollback, données absentes,
artifact autonome utilisable avant/après échec DB. Documentation des commandes locales.
