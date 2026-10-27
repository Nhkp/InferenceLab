# Prompt 11 — Après MVP : registry léger et découverte de révisions

Ajoute uniquement le control-plane nécessaire pour détecter les révisions de modèles suivis.

## Fichier

```yaml
models:
  - alias: qwen7b
    repo_id: Qwen/example
    revision: main
    allow_remote_code: false
```

Avant toute campagne, résoudre la référence mobile en commit immuable. `main` ne
constitue jamais une preuve de la révision exécutée. Cette étape n'est pas une
dépendance du runner ou du parcours Kaggle.

## Comportement

Interroger les metadata Hugging Face, détecter un nouveau commit/revision et créer un candidat potentiel d'expérience.

Ne pas :

- télécharger automatiquement des poids volumineux ;
- exécuter du code distant ;
- promouvoir automatiquement en production.

## Étapes explicites

```text
discovery -> validation -> optional download -> experiment -> promotion decision
```

## Persistance

Utiliser `model_revisions`.

## Tests

Mock réseau uniquement.
